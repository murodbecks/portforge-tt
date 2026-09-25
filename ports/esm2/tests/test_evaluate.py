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
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import evaluate
try:
    import numpy as np
except ImportError:
    np = None


def synthetic_corpus():
    names = [f"sp-{i:04d}" for i in range(14)]
    cases = [dict(name=n, sequence="MKV" * (i + 2), residues=3 * (i + 2), source="fixture")
             for i, n in enumerate(names)]
    cases += [dict(name="syn-max1024", sequence="A" * 1024, residues=1024, source="fixture"),
              dict(name="syn-pair", sequence="MA", residues=2, source="fixture"),
              dict(name="syn-unknown-X", sequence="MKVXZB", residues=6, source="fixture")]
    return {"schema_version": 1, "task": "fixture", "protocol": "fixture", "sources": [],
            "selection": {"smoke": names[0],
                          "bringup": dict(batch0=names[1], batch1=names[4], long="syn-max1024",
                                          short="syn-pair", unknown="syn-unknown-X"),
                          "full": names[2:] + names[0:2]},
            "cases": cases}


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.corpus = synthetic_corpus()

    def suite(self, stage):
        digest = hashlib.sha256(json.dumps(self.corpus, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        with patch.object(evaluate, "CORPUS_SHA256", digest):
            return evaluate.make_suite(self.corpus, stage, "esm2")

    def test_stages_and_masking(self):
        spec = self.suite("smoke")[0]
        self.assertEqual(len(spec["cases"]), 1)
        self.assertEqual(spec["vocab_size"], 33)
        self.assertEqual(spec["gates"], evaluate.GATES)
        bringup = self.suite("bringup")[0]
        cases = {c["name"]: c for c in bringup["cases"]}
        self.assertEqual(len(cases), 8)
        self.assertEqual(cases["reverse"]["compare_rows"], [1, 0])
        self.assertEqual(cases["long"]["sequences"], ["syn-max1024"])
        self.assertEqual(cases["unknown-residues"]["sequences"], ["syn-unknown-X"])
        full = self.suite("full")[0]
        self.assertEqual(full["timed_repeats"], 3)
        self.assertGreaterEqual(len(full["cases"]), 18)
        self.assertTrue(all(c["numerical"] for c in full["cases"]))

    def test_masked_positions_rule(self):
        self.assertEqual(evaluate.masked_positions(4), [])
        self.assertEqual(evaluate.masked_positions(22), [5, 21])
        self.assertEqual(evaluate.masked_positions(1024)[-1], 1013)

    def test_frozen_corpus_file(self):
        path = shipped_corpus(self)
        corpus = json.loads(path.read_text())
        digest = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        self.assertEqual(digest, evaluate.CORPUS_SHA256)
        evaluate.make_suite(corpus, "smoke", "esm2")
        corpus["cases"][0]["sequence"] += "A"
        with self.assertRaises(ValueError):
            evaluate.make_suite(corpus, "smoke", "esm2")

    def test_precision_and_execution(self):
        runtime = types.SimpleNamespace(bfloat16=object(), float32=object(), bfloat8_b=object(), float16=object())
        policy = dict(mode="bf16", weights="bf16", activations="bf16", accumulation="fp32", exceptions=[])
        self.assertEqual(evaluate.validate_precision("bf16", policy, runtime)["requested"], "bf16")
        with patch("evaluate.socket.gethostname", return_value="gpu-box"):
            import os
            with patch.dict(os.environ, {"PORTFORGE_REFERENCE_EXECUTION": "direct"}):
                self.assertEqual(evaluate.reference_execution(), "direct")
        with patch.dict(os.environ, {"PORTFORGE_REFERENCE_EXECUTION": "direct"}), \
             patch("evaluate.socket.gethostname", return_value="login-1"), self.assertRaises(RuntimeError):
            evaluate.reference_execution()


@unittest.skipIf(np is None, "NumPy required")
class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = types.SimpleNamespace(input=self.root / "input", oracle=self.root / "oracle", output=self.root / "output")
        for path in vars(self.args).values():
            path.mkdir()
        self.cases = [dict(name="batch", sequences=["s0", "s1"], numerical=True,
                           masked_positions=[[5, 21], [5, 21]]),
                      dict(name="single", sequences=["s0"], numerical=True,
                           masked_positions=[[5, 21]], compare_to="batch", compare_rows=[0])]
        self.strict_fixture()

    def strict_fixture(self):
        model = evaluate.model_spec("esm2")
        identity = evaluate.model_identity("esm2")
        spec = dict(version=2, stage="bringup", cases=self.cases, gates=evaluate.GATES.copy(),
                    timed_repeats=1, **identity)
        receipt = dict(model="esm2", repo_id=model["repo_id"], revision=model["revision"],
                       checkpoint_format=model["checkpoint_format"], verified_files=copy.deepcopy(model["files"]),
                       model_manifest_sha256=identity["model_manifest_sha256"], suite_sha256=evaluate.suite_hash(spec))
        self.expected, self.actual, self.entries, inputs = {}, {}, [], {}
        for case in self.cases:
            name, rows, length = case["name"], len(case["sequences"]), 40
            pattern = np.tile(np.linspace(-1, 1, 33, dtype=np.float32), (rows, length, 1))
            self.expected[name + "__logits"] = pattern
            self.expected[name + "__hidden"] = pattern[:, :, :1280] * 0 + pattern
            self.entries.append(dict(name=name, seconds=[.05], first_seconds=.1))
            inputs[name + "__input_ids"] = np.ones((rows, length), dtype=np.int64)
            inputs[name + "__attention_mask"] = np.ones((rows, length), dtype=np.int64)
            inputs[name + "__residues"] = np.full(rows, length - 2, dtype=np.int64)
        self.actual = {k: v.copy() for k, v in self.expected.items()}
        base = dict(**identity, suite_sha256=evaluate.suite_hash(spec), timing_protocol=evaluate.TIMING_PROTOCOL,
                    verified_weight_files=model["files"], checkpoint_receipt=receipt, precision={"requested": "fp32"})
        evaluate.write_json(self.args.input / "suite.json", spec)
        evaluate.write_json(self.args.input / "config.json", model["architecture"])
        evaluate.write_json(self.args.input / "checkpoint_receipt.json", receipt)
        evaluate.write_json(self.args.output / "measurements.json", {**base, "cases": self.entries})
        evaluate.write_json(self.args.oracle / "reference.json", base)
        np.savez(self.args.input / "inputs.npz", **inputs)
        np.savez(self.args.output / "actual.npz", **self.actual)
        np.savez(self.args.oracle / "oracle.npz", **self.expected)
        return spec

    def test_passing_bringup(self):
        result = evaluate.assess(self.args)
        self.assertTrue(result["passed"], result["failures"])
        self.assertFalse(result["accepted"])
        check = result["checks"][0]
        self.assertEqual(check["exact_masked_rows"], 2)
        self.assertEqual(check["residues"], 76)

    def test_logit_drift_fails(self):
        self.actual["batch__logits"][1, 7] += 5.0
        np.savez(self.args.output / "actual.npz", **self.actual)
        self.assertFalse(evaluate.assess(self.args)["passed"])

    def test_hidden_drift_fails(self):
        self.actual["single__hidden"] += .3
        np.savez(self.args.output / "actual.npz", **self.actual)
        result = evaluate.assess(self.args)
        self.assertFalse(result["passed"])

    def test_behavior_reorder_fails(self):
        self.actual["reverse-free"] = None
        # reorder semantics: mutate single's hidden to diverge from batch row 0
        pattern = self.actual["batch__hidden"].copy()
        self.actual["single__hidden"] = pattern[:1] + 0.02 * np.abs(pattern[:1]).mean()
        np.savez(self.args.output / "actual.npz", **self.actual)
        # single no longer matches its own oracle -> exact-masked or nrmse gate fires
        self.assertFalse(evaluate.assess(self.args)["passed"])

    def test_identity_tampering(self):
        path = self.args.output / "measurements.json"
        original = json.loads(path.read_text())
        for field, value in (("suite_sha256", "0" * 64), ("model_key", "other")):
            evaluate.write_json(path, {**original, field: value})
            self.assertFalse(evaluate.assess(self.args)["passed"])
            evaluate.write_json(path, original)

    def test_missing_timing_rejected(self):
        measured = json.loads((self.args.output / "measurements.json").read_text())
        measured["cases"] = measured["cases"][:1]
        evaluate.write_json(self.args.output / "measurements.json", measured)
        self.assertFalse(evaluate.assess(self.args)["passed"])


if __name__ == "__main__":
    unittest.main()
