"""Trusted evaluator regressions; no model weights, CUDA or TT devices required."""
# Run from the harness directory: python -m unittest discover -s ports/<pack>/tests
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACK), str(PACK.parents[1])]
import models  # noqa: E402

models.use_registry(PACK / "models.json")


def shipped_corpus(test):
    """The pack's frozen corpus.json; packs with large corpora ship none (pass --corpus)."""
    path = PACK / "corpus.json"
    if not path.is_file():
        test.skipTest("this pack does not ship its corpus")
    return path
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
from unittest.mock import Mock, patch

import evaluate
try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy required")
class InputPreservationTests(unittest.TestCase):
    def setUp(self):
        self.originals = {
            "past_values": np.array([[1.5, 2.0, 0.0, 4.25]], dtype=np.float32),
            "past_observed_mask": np.array([[1.0, 1.0, 0.0, 1.0]], dtype=np.float32),
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
        for transform in (lambda a: a.reshape(-1), lambda a: a.astype(np.float64)):
            observed = {key: value.copy() for key, value in self.originals.items()}
            observed["past_values"] = transform(observed["past_values"])
            with self.assertRaisesRegex(ValueError, "mutated caller input"):
                evaluate.validate_input_preservation(self.originals, observed)

    def test_missing_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "input set changed"):
            evaluate.validate_input_preservation(self.originals, {})


class PrecisionTests(unittest.TestCase):
    def temp_name(self):
        with tempfile.TemporaryDirectory() as directory:
            return directory
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
        policy["exceptions"] = ["BF16 activations and FP32 scaling preserve numerical stability"]
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

    def test_cpu_reference_is_allowed_and_recorded_not_rejected(self):
        corpus = shipped_corpus(self)
        modules = {"numpy": SimpleNamespace(), "torch": SimpleNamespace(
                        cuda=SimpleNamespace(is_available=lambda: False),
                        manual_seed=lambda seed: None, set_num_threads=lambda n: None),
                   "huggingface_hub": SimpleNamespace(
                        hf_hub_download=Mock(side_effect=RuntimeError("stop-after-device-check")))}
        with patch.dict(os.environ, {"PORTFORGE_REFERENCE_EXECUTION": "direct"}), \
             patch("evaluate.socket.gethostname", return_value="dedicated-gpu"), \
             patch.dict("sys.modules", modules), \
             patch.object(evaluate, "Progress", lambda *a, **kw: SimpleNamespace(
                 mark=lambda *a, **kw: None)), \
             self.assertRaisesRegex(RuntimeError, "stop-after-device-check"):
            with tempfile.TemporaryDirectory() as out:
                evaluate.reference(SimpleNamespace(command="reference", stage="smoke", model="chronos2",
                                                   corpus=str(corpus), out=out))

    def test_reference_pipeline_rejects_fp16_benchmark_requests(self):
        modules = {"numpy": SimpleNamespace(), "torch": SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            manual_seed=lambda seed: None, set_num_threads=lambda n: None)}
        with patch.dict("sys.modules", modules), \
             patch.object(evaluate, "reference_execution", return_value="direct"), \
             self.assertRaisesRegex(ValueError, "float32/bfloat16"):
            evaluate.reference(SimpleNamespace(command="reference-benchmark", precision="fp16",
                                               input="i", oracle="o", output="out", model="chronos2"))


class SuiteTests(unittest.TestCase):
    KEYS = ("etth1_hufl", "etth1_hull", "etth1_mufl", "etth1_mull",
            "etth1_lufl", "etth1_lull", "etth1_ot",
            "syn_sine_mixed", "syn_arcsinh_stress", "syn_constant")

    @classmethod
    def synthetic_corpus(cls):
        series = {key: [float((j * 37 + i * 11) % 97) / 7.0 for j in range(1536)]
                  for i, key in enumerate(cls.KEYS)}
        return {"schema_version": 1, "task": "synthetic fixture",
                "protocol": "synthetic fixture", "sources": [],
                "series": series}

    def setUp(self):
        self.corpus = self.synthetic_corpus()

    def suite(self, stage, corpus=None):
        corpus = corpus or self.corpus
        digest = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        with patch.object(evaluate, "CORPUS_SHA256", digest):
            return evaluate.make_suite(corpus, stage, "chronos2")

    def test_frozen_corpus_file_matches_pinned_hash(self):
        path = shipped_corpus(self)
        corpus = json.loads(path.read_text())
        digest = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        self.assertEqual(digest, evaluate.CORPUS_HASH if hasattr(evaluate, "CORPUS_HASH")
                         else evaluate.CORPUS_SHA256)
        evaluate.make_suite(corpus, "smoke", "chronos2")

    def test_suite_identity_rejects_tampering_and_missing_version(self):
        spec, _ = self.suite("smoke")
        for field, value in (("model_key", "other"), ("revision", "0" * 40),
                             ("model_manifest_sha256", "0" * 64), ("gates", {}), ("version", None)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate.suite_model({**spec, field: value})

    def test_receipt_binds_final_prepared_suite_and_weights(self):
        spec, _ = self.suite("smoke")
        before = evaluate.suite_hash(spec)
        spec["cases"][0]["context"] = 128
        self.assertNotEqual(before, evaluate.suite_hash(spec))
        model, identity = evaluate.suite_model(spec)
        receipt = dict(model="chronos2", repo_id=model["repo_id"], revision=model["revision"],
                       checkpoint_format=model["checkpoint_format"], verified_files=copy.deepcopy(model["files"]),
                       model_manifest_sha256=identity["model_manifest_sha256"], suite_sha256=evaluate.suite_hash(spec))
        evaluate.validate_receipt(receipt, identity, model, spec)
        with self.assertRaises(ValueError):
            evaluate.validate_receipt({**receipt, "suite_sha256": before}, identity, model, spec)
        del receipt["verified_files"][model["weight_files"][-1]]
        with self.assertRaises(ValueError):
            evaluate.validate_receipt(receipt, identity, model, spec)

    def test_masked_context_and_series_windows(self):
        corpus = self.corpus
        past, future = evaluate.series_window(corpus["series"]["etth1_hufl"], 512, 64, 0)
        self.assertEqual(len(past), 512)
        self.assertEqual(future, corpus["series"]["etth1_hufl"][512:512 + 64])
        context, mask = evaluate.masked_context(past, {"tail": 16, "holes": [3, 500]})
        self.assertEqual(len(context), 512)
        self.assertEqual(sum(mask), 512 - 17)
        self.assertEqual(context[3], 0.0)
        self.assertEqual(context[500], 0.0)
        self.assertEqual(context[495], past[495])
        self.assertTrue(all(context[i] == 0.0 for i in range(496, 512)))
        with self.assertRaises(ValueError):
            evaluate.series_window(corpus["series"]["etth1_hufl"], 512, 64, 2)

    def test_smoke_uses_actual_pinned_model_and_short_numerical_request(self):
        spec, labels = self.suite("smoke")
        self.assertEqual(spec["model"], "amazon/chronos-2")
        self.assertEqual(len(spec["revision"]), 40)
        self.assertEqual(len(spec["weight_sha256"]), 64)
        self.assertEqual(spec["gates"], evaluate.GATES)
        self.assertEqual(spec["timed_repeats"], 1)
        self.assertEqual(len(spec["cases"]), 1)
        case = spec["cases"][0]
        self.assertTrue(case["numerical"])
        self.assertEqual((case["context"], case["horizon"]), (512, 64))
        self.assertEqual(len(labels[case["name"]]), 1)
        self.assertEqual(len(labels[case["name"]][0]), 64)
        self.assertEqual(spec["quantiles"], list(evaluate.QUANTILES))

    def test_bringup_coverage_and_comparisons(self):
        spec, _ = self.suite("bringup")
        cases = {c["name"]: c for c in spec["cases"]}
        self.assertEqual(len(cases), 8)
        self.assertEqual(cases["masked-tail"]["mask"], {"tail": 16})
        self.assertEqual(cases["reverse"]["compare_rows"], [1, 0])
        self.assertEqual(cases["single0"]["compare_rows"], [0])
        self.assertEqual({c["context"] for c in cases.values()}, {64, 65, 512})
        for case in cases.values():
            self.assertTrue(case["numerical"])
            self.assertLessEqual(case["horizon"], 64)
            if "compare_to" in case:
                self.assertEqual(case["series"],
                                 [cases[case["compare_to"]]["series"][i] for i in case["compare_rows"]])

    def test_full_coverage_held_out_futures_and_revisit(self):
        spec, labels = self.suite("full")
        quality = [c for c in spec["cases"] if c.get("quality")]
        self.assertGreaterEqual(len(quality), 25)
        self.assertEqual({c["context"] for c in spec["cases"]}, {64, 128, 512})
        self.assertEqual({c["horizon"] for c in spec["cases"]}, {16, 64})
        for case in spec["cases"]:
            future = labels[case["name"]][0]
            series = self.corpus["series"][case["series"][0]]
            start = case["window"] * (case["context"] + case["horizon"]) + case["context"]
            self.assertEqual(future, series[start:start + case["horizon"]])
        self.assertTrue(any(c["name"] == "revisit" and "compare_to" in c for c in spec["cases"]))
        self.assertEqual(spec["gates"], evaluate.GATES)
        self.assertEqual(spec["timed_repeats"], 3)

    def test_windows_within_full_stage_are_disjoint(self):
        spec, _ = self.suite("full")
        per_series = {}
        for case in spec["cases"]:
            if "compare_to" in case:
                continue
            if not case["name"].endswith(("-w0", "-w1", "-w2")):
                continue
            for key in case["series"]:
                start = case["window"] * (case["context"] + case["horizon"]) + case["context"]
                per_series.setdefault(key, []).append((start, start + case["horizon"]))
        self.assertTrue(per_series)
        for key, spans in per_series.items():
            spans.sort()
            for (_, end), (start2, _) in zip(spans, spans[1:]):
                self.assertLessEqual(end, start2)

    def test_changed_corpus_rejected(self):
        with self.assertRaises(ValueError):
            evaluate.make_suite(self.corpus, "smoke", "chronos2")

    def test_atomic_progress_and_bounded_deadlines(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as log:
            progress = evaluate.Progress(directory, "smoke")
            first = progress.mark("open_device", initializing=True)
            self.assertEqual(first["timeout_seconds"], 600)
            event = progress.mark("predict", "length-65", 0)
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
        self.cases = [dict(name="batch", series=["s0", "s1"], context=512, horizon=16, numerical=True),
                      dict(name="single", series=["s0"], context=512, horizon=16, numerical=True,
                           compare_to="batch", compare_rows=[0])]
        self.stage = "bringup"
        self.expected, self.actual = {}, {}
        self.entries = []
        self.initialize_arrays()

    def initialize_arrays(self):
        self.expected, self.actual, self.entries = {}, {}, []
        for case in self.cases:
            name, rows = case["name"], len(case["series"])
            pattern = np.linspace(-1, 1, case["horizon"] * 21, dtype=np.float32).reshape(1, case["horizon"], 21)
            self.expected[name + "__quantiles"] = np.tile(pattern, (rows, 1, 1))
            self.entries.append(dict(name=name, seconds=[.1], first_seconds=.2))
        self.actual = {k: v.copy() for k, v in self.expected.items()}

    def run_assess(self):
        self.prepare_files()
        return evaluate.assess(self.args)

    def strict_fixture(self):
        self.initialize_arrays()
        for case in self.entries:
            case["first_seconds"] = .2
        return self.prepare_files()

    def prepare_files(self):
        model = evaluate.model_spec("chronos2")
        identity = evaluate.model_identity("chronos2")
        spec = dict(version=2, stage=self.stage, cases=self.cases, gates=evaluate.GATES.copy(),
                    quantiles=list(evaluate.QUANTILES), timed_repeats=1, **identity)
        receipt = dict(model="chronos2", repo_id=model["repo_id"], revision=model["revision"],
                       checkpoint_format=model["checkpoint_format"], verified_files=copy.deepcopy(model["files"]),
                       model_manifest_sha256=identity["model_manifest_sha256"], suite_sha256=evaluate.suite_hash(spec))
        inputs = {}
        for case in self.cases:
            rows, name = len(case["series"]), case["name"]
            inputs[name + "__past_values"] = np.ones((rows, case["context"]), dtype=np.float32)
            inputs[name + "__past_observed_mask"] = np.ones((rows, case["context"]), dtype=np.float32)
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

    def test_strict_fixture_reports_matched_shapes_and_raw_timings(self):
        spec = self.strict_fixture()
        result = evaluate.assess(self.args)
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["model_key"], "chronos2")
        self.assertEqual(result["suite_sha256"], evaluate.suite_hash(spec))
        self.assertEqual(result["timing_protocol"], evaluate.TIMING_PROTOCOL)
        check = result["checks"][0]
        self.assertEqual(check["forecast_points"], 2 * 16)
        self.assertEqual(check["forecasts_per_second"], 2 / .1)
        self.assertEqual(check["seconds"], [.1])
        self.assertEqual(check["first_seconds"], .2)
        self.assertIn("quantile_crossings", check)

    def test_strict_stale_suite_and_wrong_weight_metadata_fail(self):
        self.strict_fixture()
        path = self.args.output / "measurements.json"
        original = json.loads(path.read_text())
        for field, value in (("model_key", "other"), ("suite_sha256", "0" * 64),
                             ("verified_weight_files", {}), ("timing_protocol", "kernel-only")):
            evaluate.write_json(path, {**original, field: value})
            result = evaluate.assess(self.args)
            self.assertFalse(result["passed"])
            self.assertTrue(any("identity" in failure for failure in result["failures"]))

    def test_strict_config_and_both_tensor_shapes_are_checked(self):
        for mode in ("quantiles", "config"):
            self.strict_fixture()
            if mode == "config":
                path = self.args.input / "config.json"
                config = json.loads(path.read_text())
                config["d_model"] = 12
                evaluate.write_json(path, config)
            else:
                self.actual["batch__quantiles"] = np.ones((2, 2, 21), dtype=np.float32)
                np.savez(self.args.output / "actual.npz", **self.actual)
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
        self.actual["batch__quantiles"][0] += .5
        result = self.run_assess()
        self.assertFalse(result["passed"])
        self.assertIn("NRMSE", " ".join(result["failures"]))

    def test_nonfinite_wrong_shape_missing_and_extra_arrays_fail(self):
        for mode in ("nan", "shape", "missing", "extra"):
            self.initialize_arrays()
            if mode == "nan":
                self.actual["batch__quantiles"][0, 0, 0] = np.nan
            elif mode == "shape":
                self.actual["batch__quantiles"] = np.ones((1, 16, 21), dtype=np.float32)
            elif mode == "missing":
                del self.actual["single__quantiles"]
            else:
                self.actual["unexpected"] = np.ones(1)
            with self.subTest(mode=mode):
                self.assertFalse(self.run_assess()["passed"])

    def test_batch_single_semantic_change_fails_even_with_good_numerics(self):
        pattern = self.expected["batch__quantiles"][0].astype(np.float64)
        # per-row NRMSE ~0.02: above the 0.01 behavior gate, below the 0.04 oracle gate
        self.actual["batch__quantiles"][0] = (pattern + 0.02 * np.sqrt(np.mean(pattern ** 2))).astype(np.float32)
        result = self.run_assess()
        self.assertFalse(result["passed"])
        self.assertIn("semantics", " ".join(result["failures"]))

    def test_invalid_timing_evidence_rejected(self):
        for mode in ("missing_timing", "duplicate_timing", "zero_time"):
            self.initialize_arrays()
            if mode == "missing_timing":
                self.entries.pop()
            elif mode == "duplicate_timing":
                self.entries.append(self.entries[0])
            else:
                self.entries[0]["seconds"] = [0]
            with self.subTest(mode=mode):
                self.assertFalse(self.run_assess()["passed"])

    def full_fixture(self, inflation=0.0):
        self.stage = "full"
        self.cases = [dict(name="w0", series=["s0"], context=512, horizon=64,
                           numerical=True, quality=True),
                      dict(name="w1", series=["s1"], context=128, horizon=16,
                           numerical=True, quality=True)]
        self.initialize_arrays()
        rng = np.random.default_rng(7)
        labels = {}
        for case in self.cases:
            name = case["name"]
            future = rng.normal(size=(1, case["horizon"])).astype(np.float64)
            base = self.expected[name + "__quantiles"].astype(np.float64)
            self.actual[name + "__quantiles"] = (base + inflation * 3.0).astype(np.float32)
            labels[name] = future.tolist()
        evaluate.write_json(self.args.oracle / "labels.json", labels)

    def test_full_gates_weighted_quantile_loss_inflation(self):
        for inflation, passed in ((0.0, True), (.001, True), (.5, False)):
            self.full_fixture(inflation)
            result = self.run_assess()
            self.assertEqual(result["accepted"], passed, result["failures"])
            self.assertEqual(len(result["quality"]), 2 if passed else 0)

    def test_full_requires_labels_for_every_quality_case(self):
        self.full_fixture()
        labels = json.loads((self.args.oracle / "labels.json").read_text())
        del labels["w1"]
        evaluate.write_json(self.args.oracle / "labels.json", labels)
        result = self.run_assess()
        self.assertFalse(result["passed"])

    def test_pinball_loss_and_nrmse_math(self):
        quantiles = np.array([[[1.0, 2.0, 3.0]]])  # [1 batch, 1 step, 3 levels]
        future = np.array([[2.5]])
        loss = evaluate.pinball_loss(quantiles, future, [0.1, 0.5, 0.9])
        # tau*(y-q) when y>=q, (1-tau)*(q-y) otherwise: .1*1.5 + .5*.5 + .1*.5; mean/3, / mean|y|
        expected = ((0.1 * 1.5 + 0.5 * 0.5 + 0.1 * 0.5) / 3) / 2.5
        self.assertAlmostEqual(loss, expected, places=9)
        a = np.array([[[1.0, 2.0]]])
        b = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])[:1]
        self.assertEqual(evaluate.quantile_nrmse(a, b), 0.0)
        with self.assertRaises(ValueError):
            evaluate.quantile_nrmse(np.array([[[np.inf, 1.0]]]), b)


if __name__ == "__main__":
    unittest.main()
