"""Trusted evaluator regressions; no model weights, CUDA or TT devices required."""
# Run from the harness directory: python -m unittest discover -s ports/<pack>/tests
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACK), str(PACK.parents[1])]
import models  # noqa: E402

models.use_registry(PACK / "models.json")
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluate
try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy required")
class InputPreservationTests(unittest.TestCase):
    def setUp(self):
        self.originals = {
            "input_ids": np.array([[3, 4, 1]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 0]], dtype=np.int64),
            "decoder_input_ids": np.array([[2, 5]], dtype=np.int64),
        }

    def test_unchanged_inputs_and_readonly_views_pass(self):
        observed = {name: value.copy() for name, value in self.originals.items()}
        for value in observed.values():
            value.flags.writeable = False
        evaluate.validate_input_preservation(self.originals, observed)

    def test_mutating_any_caller_array_fails_even_if_results_are_correct(self):
        for name in self.originals:
            with self.subTest(name=name):
                observed = {key: value.copy() for key, value in self.originals.items()}
                observed[name][0, 0] += 1
                with self.assertRaisesRegex(ValueError, "mutated caller input: " + name):
                    evaluate.validate_input_preservation(self.originals, observed)

    def test_shape_and_dtype_changes_cannot_hide_equal_values(self):
        for transform in (lambda a: a.reshape(-1), lambda a: a.astype(np.int32)):
            observed = {key: value.copy() for key, value in self.originals.items()}
            observed["input_ids"] = transform(observed["input_ids"])
            with self.assertRaisesRegex(ValueError, "mutated caller input"):
                evaluate.validate_input_preservation(self.originals, observed)

    def test_missing_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "input set changed"):
            evaluate.validate_input_preservation(self.originals, {})


class PrecisionTests(unittest.TestCase):
    def policy(self, mode="bf16", **changes):
        return dict(mode=mode, weights=mode, activations=mode, accumulation="fp32", exceptions=[], **changes)

    def test_bf16_fp32_and_block_float_are_distinct_declarations(self):
        runtime = SimpleNamespace(bfloat16=object(), float32=object(), bfloat8_b=object())
        for mode in ("bf16", "fp32", "bfp8_b"):
            result = evaluate.validate_precision(mode, self.policy(mode), runtime)
            self.assertEqual(result["requested"], mode)
            self.assertEqual(result["effective"]["weights"], mode)
            self.assertTrue(result["declaration_only"])
        self.assertIn("not IEEE FP8", evaluate.validate_precision("bfp8_b", self.policy("bfp8_b"))["format_note"])

    def test_unsupported_native_fp16_and_true_fp8_do_not_silently_substitute(self):
        for mode in ("fp16", "fp8"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                evaluate.validate_precision(mode, self.policy(mode), SimpleNamespace(bfloat16=object()))
        supported = evaluate.validate_precision("fp16", self.policy("fp16"), SimpleNamespace(float16=object()))
        self.assertEqual(supported["effective"]["mode"], "fp16")

    def test_mixed_precision_requires_explicit_exceptions(self):
        policy = self.policy("bfp8_b")
        policy["activations"] = "bf16"
        with self.assertRaises(ValueError):
            evaluate.validate_precision("bfp8_b", policy)
        policy["exceptions"] = ["BF16 activations and FP32 normalization preserve numerical stability"]
        self.assertEqual(evaluate.validate_precision("bfp8_b", policy)["effective"], policy)

    def test_mislabeled_or_missing_policies_are_rejected(self):
        policies = [None, {}, self.policy("fp32")]
        for field, value in (("weights", "fp32"), ("accumulation", ""), ("exceptions", "none"),
                             ("exceptions", [""]), ("activations", "fp8")):
            policy = self.policy()
            policy[field] = value
            policies.append(policy)
        for policy in policies:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                evaluate.validate_precision("bf16", policy)

    def test_direct_gpu_host_is_explicit_and_slurm_requires_allocation(self):
        with patch("evaluate.socket.gethostname", return_value="dedicated-gpu"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                evaluate.reference_execution()
            with patch.dict(os.environ, {"PORTFORGE_REFERENCE_EXECUTION": "direct"}):
                self.assertEqual(evaluate.reference_execution(), "direct")
            with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}):
                self.assertEqual(evaluate.reference_execution(), "slurm")

    def test_login_nodes_and_unknown_execution_modes_are_rejected(self):
        for mode, host in (("direct", "login-student-lab"), ("slurm", "lo-login"), ("unknown", "gpu")):
            with patch.dict(os.environ, {"PORTFORGE_REFERENCE_EXECUTION": mode, "SLURM_JOB_ID": "123"}), \
                 patch("evaluate.socket.gethostname", return_value=host), self.assertRaises(RuntimeError):
                evaluate.reference_execution()

    def test_direct_mode_still_refuses_non_cuda_execution(self):
        modules = {"numpy": SimpleNamespace(), "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                   "transformers": SimpleNamespace(AutoModelForSeq2SeqLM=None, AutoTokenizer=None)}
        with patch.dict(os.environ, {"PORTFORGE_REFERENCE_EXECUTION": "direct"}), \
             patch("evaluate.socket.gethostname", return_value="dedicated-gpu"), \
             patch.dict("sys.modules", modules), self.assertRaisesRegex(RuntimeError, "CUDA required"):
            evaluate.reference(SimpleNamespace())


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.corpus = {lang: [f"{lang} row {i}" for i in range(1012)]
                       for lang in ("eng_Latn", *evaluate.LANGUAGES)}

    def suite(self, stage, model="600m"):
        digest = hashlib.sha256(json.dumps(self.corpus, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with patch.object(evaluate, "CORPUS_SHA256", digest):
            return evaluate.make_suite(self.corpus, stage, model)

    def test_model_variants_preserve_cases_but_have_distinct_pinned_identities(self):
        cases = self.suite("bringup")[0]["cases"]
        identities = set()
        for name in ("600m", "1.3b-distilled", "3.3b"):
            spec, labels = self.suite("bringup", name)
            self.assertEqual(spec["version"], 2)
            self.assertEqual(spec["model_key"], name)
            self.assertEqual(spec["cases"], cases)
            model, identity = evaluate.suite_model(spec)
            self.assertEqual(identity["revision"], model["revision"])
            identities.add(identity["model_manifest_sha256"])
        self.assertEqual(len(identities), 3)

    def test_suite_identity_rejects_cross_model_tampering_and_missing_version(self):
        spec, _ = self.suite("smoke", "1.3b-distilled")
        for field, value in (("model_key", "600m"), ("revision", "0" * 40),
                             ("model_manifest_sha256", "0" * 64), ("gates", {}), ("version", None)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate.suite_model({**spec, field: value})

    def test_receipt_binds_final_prepared_suite_and_every_shard(self):
        spec, _ = self.suite("smoke", "3.3b")
        before = evaluate.suite_hash(spec)
        spec["cases"][0]["target_id"] = 10
        self.assertNotEqual(before, evaluate.suite_hash(spec))
        model, identity = evaluate.suite_model(spec)
        receipt = dict(model="3.3b", repo_id=model["repo_id"], revision=model["revision"],
                       checkpoint_format=model["checkpoint_format"], verified_files=copy.deepcopy(model["files"]),
                       model_manifest_sha256=identity["model_manifest_sha256"], suite_sha256=evaluate.suite_hash(spec))
        evaluate.validate_receipt(receipt, identity, model, spec)
        with self.assertRaises(ValueError):
            evaluate.validate_receipt({**receipt, "suite_sha256": before}, identity, model, spec)
        del receipt["verified_files"][model["weight_files"][-1]]
        with self.assertRaises(ValueError):
            evaluate.validate_receipt(receipt, identity, model, spec)

    def test_token_validation_uses_selected_architecture(self):
        config = dict(decoder_start_token_id=7, eos_token_id=8, pad_token_id=9, vocab_size=20)
        self.assertEqual(evaluate.canonical_tokens([7, 10, 11, 8, 9], 10, 4, config), [7, 10, 11, 8])
        for row in ([2, 10, 11, 8, 9], [7, 10, 20, 8], [7, 10, 9, 8]):
            with self.assertRaises(ValueError):
                evaluate.canonical_tokens(row, 10, 4, config)

    def test_single_file_weights_are_hashed_without_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-filename.bin"
            path.write_bytes(b"fake checkpoint")
            model = copy.deepcopy(evaluate.model_spec("600m"))
            filename = model["weight_files"][0]
            model["files"][filename] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            with patch.object(evaluate.portforge_eval, "model_spec", return_value=model):
                self.assertEqual(evaluate.verify_weights(path, "600m"), {filename: model["files"][filename]})
                path.write_bytes(b"fake checkpoins")
                with self.assertRaises(ValueError):
                    evaluate.verify_weights(path, "600m")

    def test_sharded_directory_checks_index_and_every_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = copy.deepcopy(evaluate.model_spec("3.3b"))
            payloads = {filename: ("fixture " + filename).encode() for filename in model["weight_files"]}
            payloads["config.json"] = json.dumps(model["architecture"]).encode()
            payloads[model["weight_index"]] = json.dumps({"weight_map": {
                "layer" + str(i): filename for i, filename in enumerate(model["weight_files"])}}).encode()
            model["files"] = {filename: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                              for filename, data in payloads.items()}
            for filename, data in payloads.items():
                (root / filename).write_bytes(data)
            with patch.object(evaluate, "model_spec", return_value=model), patch("models.model_spec", return_value=model):
                self.assertEqual(evaluate.verify_weights(root, "3.3b"), model["files"])
                (root / model["weight_files"][-1]).unlink()
                with self.assertRaises(ValueError):
                    evaluate.verify_weights(root, "3.3b")
                with self.assertRaises(ValueError):
                    evaluate.verify_weights(root / model["weight_files"][0], "3.3b")

    def test_smoke_uses_actual_pinned_model_and_short_numerical_request(self):
        spec, labels = self.suite("smoke")
        self.assertEqual(spec["model"], "facebook/nllb-200-distilled-600M")
        self.assertEqual(len(spec["revision"]), 40)
        self.assertEqual(len(spec["weight_sha256"]), 64)
        self.assertEqual(spec["gates"], evaluate.GATES)
        self.assertEqual(spec["timed_repeats"], 1)
        self.assertEqual(len(spec["cases"]), 1)
        self.assertTrue(spec["cases"][0]["numerical"])
        self.assertEqual(spec["cases"][0]["max_new_tokens"], 4)
        self.assertEqual(labels, {})

    def test_bringup_coverage_and_comparisons(self):
        spec, _ = self.suite("bringup")
        cases = {c["name"]: c for c in spec["cases"]}
        self.assertEqual(len(cases), 8)
        self.assertEqual({c["token_length"] for c in cases.values() if "token_length" in c}, {32, 33})
        self.assertEqual(cases["padding"]["extra_padding"], 32)
        self.assertEqual(cases["reverse"]["compare_rows"], [1, 0])
        self.assertEqual(cases["single0"]["compare_rows"], [0])
        self.assertIn("", cases["unicode-empty"]["texts"])
        for case in cases.values():
            self.assertTrue(case["numerical"])
            self.assertLessEqual(case["max_new_tokens"], 8)
            if "compare_to" in case:
                self.assertEqual(case["texts"], [cases[case["compare_to"]]["texts"][i] for i in case["compare_rows"]])

    def test_full_coverage_human_labels_private_and_cache_revisit_order(self):
        spec, labels = self.suite("full")
        quality = [c for c in spec["cases"] if c.get("quality")]
        self.assertEqual(sum(len(c["texts"]) for c in quality), 8096)
        self.assertEqual(sum(len(x) for x in labels.values()), 8096)
        self.assertEqual(len({(c["src"], c["tgt"]) for c in quality}), 8)
        self.assertNotIn("references", json.dumps(spec))
        self.assertEqual({c["token_length"] for c in spec["cases"] if "token_length" in c},
                         {2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256})
        revisits = spec["cases"][-8:]
        self.assertTrue(all(c["name"].endswith("-revisit") for c in revisits))
        self.assertEqual(spec["gates"], evaluate.GATES)

    def test_changed_corpus_rejected(self):
        with self.assertRaises(ValueError):
            evaluate.make_suite(self.corpus, "smoke")

    def test_token_contract(self):
        self.assertEqual(evaluate.canonical_tokens([2, 10, 40, 2, 1], 10, 4), [2, 10, 40, 2])
        self.assertEqual(evaluate.canonical_tokens([2, 10, 40], 10, 2), [2, 10, 40])
        for row in ([2, 11, 40, 2], [2, 10, 40], [2, 10, 1, 2], [2, 10, 2, 40], [2, 10, 999999, 2]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                evaluate.canonical_tokens(row, 10, 4)

    def test_atomic_progress_and_bounded_deadlines(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as log:
            progress = evaluate.Progress(directory, "smoke")
            first = progress.mark("open_device", initializing=True)
            self.assertEqual(first["timeout_seconds"], 600)
            event = progress.mark("generate", "length-33", 0)
            self.assertEqual(event["timeout_seconds"], 180)
            self.assertEqual(event["sequence"], 2)
            self.assertEqual(event["repeat"], 0)
            self.assertEqual(json.loads((Path(directory) / "progress.json").read_text()), event)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
            self.assertEqual(json.loads(log.getvalue().splitlines()[-1]), {"progress": event})


@unittest.skipIf(np is None, "NumPy required for independent artifact fixtures")
class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(input=self.root / "input", oracle=self.root / "oracle", output=self.root / "output")
        for path in vars(self.args).values():
            path.mkdir()
        self.cases = [dict(name="batch", texts=["a", "b"], target_id=10, max_new_tokens=4, numerical=True),
                      dict(name="single", texts=["a"], target_id=10, max_new_tokens=4, numerical=True,
                           compare_to="batch", compare_rows=[0])]
        self.stage = "bringup"
        self.expected, self.actual = {}, {}
        self.entries = []
        self.initialize_arrays()

    def initialize_arrays(self):
        self.expected, self.actual, self.entries = {}, {}, []
        for case in self.cases:
            name, rows = case["name"], len(case["texts"])
            self.expected[name + "__tokens"] = np.tile([2, 10, 40, 2], (rows, 1))
            self.expected[name + "__encoder"] = np.ones((rows, 2, 2), dtype=np.float32)
            self.expected[name + "__logits"] = np.ones((rows, 4), dtype=np.float32)
            self.entries.append(dict(name=name, seconds=[.1]))
        self.actual = {k: v.copy() for k, v in self.expected.items()}

    def run_assess(self):
        evaluate.write_json(self.args.input / "suite.json", dict(version=1, stage=self.stage, cases=self.cases))
        evaluate.write_json(self.args.output / "measurements.json", dict(cases=self.entries))
        np.savez(self.args.output / "actual.npz", **self.actual)
        np.savez(self.args.oracle / "oracle.npz", **self.expected)
        return evaluate.assess(self.args)

    def strict_fixture(self, model_name="1.3b-distilled"):
        model = evaluate.model_spec(model_name)
        identity = evaluate.model_identity(model_name)
        spec = dict(version=2, stage="bringup", cases=self.cases, gates=evaluate.GATES.copy(),
                    timed_repeats=1, **identity)
        receipt = dict(model=model_name, repo_id=model["repo_id"], revision=model["revision"],
                       checkpoint_format=model["checkpoint_format"], verified_files=copy.deepcopy(model["files"]),
                       model_manifest_sha256=identity["model_manifest_sha256"], suite_sha256=evaluate.suite_hash(spec))
        self.initialize_arrays()
        inputs = {}
        for case in self.cases:
            rows, name = len(case["texts"]), case["name"]
            self.expected[name + "__encoder"] = np.ones((rows, 2, model["architecture"]["d_model"]), dtype=np.float32)
            self.expected[name + "__logits"] = np.ones((rows, model["architecture"]["vocab_size"]), dtype=np.float32)
            inputs[name + "__input_ids"] = np.ones((rows, 2), dtype=np.int64)
        self.actual = {k: v.copy() for k, v in self.expected.items()}
        for case in self.entries:
            case["first_seconds"] = .2
        meta = dict(**identity, suite_sha256=evaluate.suite_hash(spec), timing_protocol=evaluate.TIMING_PROTOCOL,
                    verified_weight_files=model["files"], checkpoint_receipt=receipt, precision={"requested": "fp32"}, cases=self.entries)
        evaluate.write_json(self.args.input / "suite.json", spec)
        evaluate.write_json(self.args.input / "config.json", model["architecture"])
        evaluate.write_json(self.args.input / "checkpoint_receipt.json", receipt)
        evaluate.write_json(self.args.output / "measurements.json", meta)
        evaluate.write_json(self.args.oracle / "reference.json", meta)
        np.savez(self.args.input / "inputs.npz", **inputs)
        np.savez(self.args.output / "actual.npz", **self.actual)
        np.savez(self.args.oracle / "oracle.npz", **self.expected)
        return spec

    def test_strict_larger_models_report_matched_shapes_tokens_and_raw_timings(self):
        for name in ("1.3b-distilled", "3.3b"):
            spec = self.strict_fixture(name)
            result = evaluate.assess(self.args)
            self.assertTrue(result["passed"], result["failures"])
            self.assertEqual(result["model_key"], name)
            self.assertEqual(result["suite_sha256"], evaluate.suite_hash(spec))
            self.assertEqual(result["timing_protocol"], evaluate.TIMING_PROTOCOL)
            check = result["checks"][0]
            self.assertEqual(check["generated_tokens"], 6)
            self.assertEqual(check["tokens_per_second"], 60)
            self.assertEqual(check["seconds"], [.1])
            self.assertEqual(check["first_seconds"], .2)

    def test_strict_cross_model_stale_suite_and_wrong_weight_metadata_fail(self):
        self.strict_fixture("3.3b")
        path = self.args.output / "measurements.json"
        original = json.loads(path.read_text())
        for field, value in (("model_key", "600m"), ("suite_sha256", "0" * 64),
                             ("verified_weight_files", {}), ("timing_protocol", "kernel-only")):
            evaluate.write_json(path, {**original, field: value})
            result = evaluate.assess(self.args)
            self.assertFalse(result["passed"])
            self.assertTrue(any("identity" in failure for failure in result["failures"]))

    def test_strict_config_and_both_oracle_candidate_tensor_shapes_are_checked(self):
        for mode in ("encoder", "logits", "config"):
            self.strict_fixture("3.3b")
            if mode == "config":
                path = self.args.input / "config.json"
                config = json.loads(path.read_text())
                config["decoder_layers"] = 12
                evaluate.write_json(path, config)
            else:
                name = "batch__" + mode
                self.actual[name] = self.expected[name] = np.ones((2, 2, 1024) if mode == "encoder" else (2, 4), dtype=np.float32)
                np.savez(self.args.output / "actual.npz", **self.actual)
                np.savez(self.args.oracle / "oracle.npz", **self.expected)
            self.assertFalse(evaluate.assess(self.args)["passed"])

    def test_strict_reference_precision_and_timing_repetitions_cannot_change(self):
        self.strict_fixture()
        path = self.args.oracle / "reference.json"
        reference = json.loads(path.read_text())
        evaluate.write_json(path, {**reference, "precision": {"requested": "bf16"}})
        self.assertFalse(evaluate.assess(self.args)["passed"])
        evaluate.write_json(path, reference)
        path = self.args.output / "measurements.json"
        measured = json.loads(path.read_text())
        measured["cases"][0]["seconds"] = [.1, .1]
        evaluate.write_json(path, measured)
        self.assertFalse(evaluate.assess(self.args)["passed"])
        measured["cases"][0]["seconds"] = [.1]
        measured["cases"][0]["first_seconds"] = -1
        evaluate.write_json(path, measured)
        self.assertFalse(evaluate.assess(self.args)["passed"])

    def test_passing_short_stage_is_not_final_acceptance(self):
        result = self.run_assess()
        self.assertTrue(result["passed"])
        self.assertFalse(result["accepted"])
        self.assertFalse(result["production_certified"])
        self.assertEqual(result["max_nrmse"], 0)
        self.assertEqual(result["case_count"], 2)

    def test_precision_identity_survives_independent_assessment(self):
        self.run_assess()
        path = self.args.output / "measurements.json"
        measured = json.loads(path.read_text())
        measured["precision"] = evaluate.validate_precision("bf16", PrecisionTests().policy())
        evaluate.write_json(path, measured)
        result = evaluate.assess(self.args)
        self.assertEqual(result["precision"]["requested"], "bf16")
        self.assertEqual(result["precision"]["effective"]["accumulation"], "fp32")

    def test_one_bad_row_cannot_hide_in_batch_average(self):
        self.actual["batch__encoder"][0] += .05
        result = self.run_assess()
        self.assertFalse(result["passed"])
        self.assertIn("NRMSE", " ".join(result["failures"]))

    def test_nonfinite_wrong_shape_missing_and_extra_arrays_fail(self):
        for mode in ("nan", "shape", "missing", "extra"):
            self.initialize_arrays()
            if mode == "nan":
                self.actual["batch__logits"][0, 0] = np.nan
            elif mode == "shape":
                self.actual["batch__logits"] = np.ones((1, 4))
            elif mode == "missing":
                del self.actual["batch__encoder"]
            else:
                self.actual["unexpected"] = np.ones(1)
            with self.subTest(mode=mode):
                self.assertFalse(self.run_assess()["passed"])

    def test_batch_single_semantic_change_fails_even_with_good_numerics(self):
        self.actual["single__tokens"][0, 2] = 41
        result = self.run_assess()
        self.assertFalse(result["passed"])
        self.assertIn("semantics", " ".join(result["failures"]))

    def test_invalid_tokens_and_timing_evidence_rejected(self):
        for mode in ("float_tokens", "bad_language", "missing_timing", "duplicate_timing", "zero_time"):
            self.initialize_arrays()
            if mode == "float_tokens":
                self.actual["batch__tokens"] = self.actual["batch__tokens"].astype(float)
            elif mode == "bad_language":
                self.actual["batch__tokens"][0, 1] = 11
            elif mode == "missing_timing":
                self.entries.pop()
            elif mode == "duplicate_timing":
                self.entries.append(self.entries[0])
            else:
                self.entries[0]["seconds"] = [0]
            with self.subTest(mode=mode):
                self.assertFalse(self.run_assess()["passed"])

    def full_fixture(self, rows=1012):
        self.stage = "full"
        self.cases = [dict(name=f"{src}-{tgt}", src=src, tgt=tgt, texts=["source"] * rows,
                           numerical=True, quality=True, target_id=10, max_new_tokens=4)
                      for src, tgt in [("eng_Latn", l) for l in evaluate.LANGUAGES] + [(l, "eng_Latn") for l in evaluate.LANGUAGES]]
        self.initialize_arrays()
        evaluate.write_json(self.args.oracle / "labels.json", {c["name"]: ["human"] * rows for c in self.cases})
        evaluate.write_json(self.args.oracle / "reference.json", dict(cases=[dict(name=c["name"], translations=["reference"] * rows) for c in self.cases]))

    def quality_modules(self, candidate_score=60):
        tokenizer = SimpleNamespace(batch_decode=lambda rows, **kw: ["candidate"] * len(rows))
        model = SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
        metric = SimpleNamespace(corpus_score=lambda rows, refs: SimpleNamespace(score=candidate_score if rows[0] == "candidate" else 60))
        return {"transformers": SimpleNamespace(AutoTokenizer=model),
                "sacrebleu.metrics": SimpleNamespace(CHRF=lambda **kw: metric)}

    def test_full_requires_all_eight_quality_directions_and_bounded_loss(self):
        self.full_fixture()
        for score, passed in ((60, True), (59.5, True), (59.49, False)):
            with patch.dict("sys.modules", self.quality_modules(score)):
                result = self.run_assess()
            self.assertEqual(result["accepted"], passed)
            self.assertEqual(len(result["quality"]), 8)
            self.assertTrue(all(q["rows"] == 1012 for q in result["quality"].values()))

    def test_full_cannot_pass_incomplete_corpus(self):
        self.full_fixture(rows=4)
        with patch.dict("sys.modules", self.quality_modules()):
            result = self.run_assess()
        self.assertFalse(result["passed"])
        self.assertFalse(result["accepted"])


if __name__ == "__main__":
    unittest.main()
