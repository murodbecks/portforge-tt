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
import base64
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch

import evaluate
try:
    import numpy as np
except ImportError:
    np = None


def synth_wav_b64(seconds=1.0, hz=440.0, rate=16000):
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(12000 * (__import__("math").sin(2 * 3.14159 * hz * t / rate))))
                               for t in range(int(seconds * rate))))
    return base64.b64encode(buf.getvalue()).decode()


def synthetic_corpus():
    names = [f"libri-{i:04d}" for i in range(15)]
    cases = [dict(name=n, transcript=f"WORDS FOR CLIP {i}", audio_b64=synth_wav_b64(2.0 + i * 0.5),
                  seconds=2.0 + i * 0.5, source="librispeech-dev-clean") for i, n in enumerate(names)]
    cases += [dict(name="syn-silence", transcript="", audio_b64=synth_wav_b64(2.0, hz=0.0), seconds=2.0, source="synthetic"),
              dict(name="syn-tone", transcript="", audio_b64=synth_wav_b64(3.0, hz=1000.0), seconds=3.0, source="synthetic")]
    return {"schema_version": 1, "task": "fixture", "protocol": "fixture", "sources": [],
            "selection": {"smoke": names[0],
                          "bringup": dict(batch0=names[1], batch1=names[4], long=names[-1], short=names[0],
                                          silence="syn-silence"),
                          "full": names[2:] + names[0:2]},
            "cases": cases}


class PrecisionTests(unittest.TestCase):
    def policy(self, mode="bf16", **changes):
        return dict(mode=mode, weights=mode, activations=mode, accumulation="fp32", exceptions=[], **changes)

    def test_modes_and_mixed_exceptions(self):
        runtime = types.SimpleNamespace(bfloat16=object(), float32=object(), bfloat8_b=object(), float16=object())
        for mode in ("bf16", "fp32", "fp16", "bfp8_b"):
            self.assertEqual(evaluate.validate_precision(mode, self.policy(mode), runtime)["requested"], mode)
        self.assertIn("not IEEE FP8", evaluate.validate_precision("bfp8_b", self.policy("bfp8_b"))["format_note"])
        policy = self.policy("bfp8_b"); policy["activations"] = "bf16"
        with self.assertRaises(ValueError):
            evaluate.validate_precision("bfp8_b", policy)
        policy["exceptions"] = ["BF16 joint network preserves argmax stability"]
        evaluate.validate_precision("bfp8_b", policy)
        with self.assertRaises(ValueError):
            evaluate.validate_precision("bf16", None)

    def test_execution_modes(self):
        with patch("evaluate.socket.gethostname", return_value="dedicated-gpu"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                evaluate.reference_execution()
            with patch.dict(os.environ, {"PORTFORGE_REFERENCE_EXECUTION": "direct"}):
                self.assertEqual(evaluate.reference_execution(), "direct")
        with patch.dict(os.environ, {"PORTFORGE_REFERENCE_EXECUTION": "direct"}), \
             patch("evaluate.socket.gethostname", return_value="login-node"), self.assertRaises(RuntimeError):
            evaluate.reference_execution()


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.corpus = synthetic_corpus()

    def suite(self, stage):
        digest = hashlib.sha256(json.dumps(self.corpus, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        with patch.object(evaluate, "CORPUS_SHA256", digest):
            return evaluate.make_suite(self.corpus, stage, "parakeet")

    def test_stages_and_gates(self):
        for stage, count in (("smoke", 1), ("bringup", 8)):
            spec, labels = self.suite(stage)
            self.assertEqual(len(spec["cases"]), count)
            self.assertEqual(spec["gates"], evaluate.GATES)
            self.assertEqual(spec["model"], "nvidia/parakeet-tdt-0.6b-v3")
            self.assertEqual(spec["timed_repeats"], 1)
        spec, labels = self.suite("full")
        self.assertGreaterEqual(len(spec["cases"]), 20)
        self.assertEqual(spec["timed_repeats"], 3)
        quality = [c for c in spec["cases"] if c.get("quality")]
        self.assertGreaterEqual(len(quality), 15)
        for case in spec["cases"]:
            self.assertEqual(len(labels[case["name"]]), len(case["clips"]))
        cases = {c["name"]: c for c in self.suite("bringup")[0]["cases"]}
        self.assertEqual(cases["reverse"]["compare_rows"], [1, 0])
        self.assertEqual(cases["single0"]["compare_to"], "batch")
        self.assertEqual(cases["silence"]["clips"], ["syn-silence"])
        self.assertTrue(all(c["numerical"] for c in spec["cases"]))

    def test_frozen_corpus_file_and_hash_rejection(self):
        path = shipped_corpus(self)
        corpus = json.loads(path.read_text())
        digest = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        self.assertEqual(digest, evaluate.CORPUS_SHA256)
        evaluate.make_suite(corpus, "smoke", "parakeet")
        corpus["cases"][0]["seconds"] = 99.0
        with self.assertRaises(ValueError):
            evaluate.make_suite(corpus, "smoke", "parakeet")

    def test_canonical_tokens_and_wer(self):
        self.assertEqual(evaluate.canonical_tokens([5, 6, 7, 2, 2], 8192, 2, 8193), [5, 6, 7])
        self.assertEqual(evaluate.canonical_tokens([5, 2, 2], 8192, 2, 8193), [5])
        self.assertEqual(evaluate.canonical_tokens([5, 6], 8192, 2, 8193), [5, 6])
        with self.assertRaises(ValueError):
            evaluate.canonical_tokens([5, 999999], 8192, 2, 8193)
        self.assertEqual(evaluate.word_error_rate("hello world", "hello world"), 0.0)
        self.assertAlmostEqual(evaluate.word_error_rate("hello there", "hello world"), 0.5)
        self.assertEqual(evaluate.word_error_rate("a b c", ""), 1.0)
        self.assertEqual(evaluate.word_error_rate("", "a b"), 1.0)

    @unittest.skipIf(np is None, "NumPy required")
    def test_encoder_nrmse(self):
        b = np.ones((2, 4, 8), dtype=np.float64)
        self.assertEqual(evaluate.quantile_free_nrmse(b.copy(), b), 0.0)
        bad = b.copy(); bad[1] += 10.0
        self.assertGreater(evaluate.quantile_free_nrmse(bad, b), 0.04)
        with self.assertRaises(ValueError):
            evaluate.quantile_free_nrmse(np.full((2, 4, 8), np.nan), b)

    @unittest.skipIf(np is None, "NumPy required")
    def test_load_audio_decodes_embedded_wav(self):
        try:
            import soundfile  # noqa: F401
        except ImportError:
            self.skipTest("soundfile required")
        data = evaluate.load_audio({"audio_b64": synth_wav_b64(0.5)})
        self.assertEqual(data.dtype, np.float32)
        self.assertEqual(len(data), 8000)


@unittest.skipIf(np is None, "NumPy required")
class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = types.SimpleNamespace(input=self.root / "input", oracle=self.root / "oracle", output=self.root / "output")
        for path in vars(self.args).values():
            path.mkdir()
        self.cases = [dict(name="batch", clips=["c0", "c1"], numerical=True),
                      dict(name="single", clips=["c0"], numerical=True, compare_to="batch", compare_rows=[0])]
        self.stage = "bringup"
        self.expected, self.actual, self.entries = {}, {}, []
        self.strict_fixture()

    def strict_fixture(self):
        model = evaluate.model_spec("parakeet")
        identity = evaluate.model_identity("parakeet")
        spec = dict(version=2, stage=self.stage, cases=self.cases, gates=evaluate.GATES.copy(),
                    timed_repeats=1, **identity)
        receipt = dict(model="parakeet", repo_id=model["repo_id"], revision=model["revision"],
                       checkpoint_format=model["checkpoint_format"], verified_files=copy.deepcopy(model["files"]),
                       model_manifest_sha256=identity["model_manifest_sha256"], suite_sha256=evaluate.suite_hash(spec))
        self.expected, self.actual, self.entries = {}, {}, []
        for case in self.cases:
            name, rows = case["name"], len(case["clips"])
            self.expected[name + "__tokens"] = np.tile([10, 11, 12, 13, 2, 2], (rows, 1))
            self.expected[name + "__encoder"] = np.tile(np.linspace(-1, 1, 32, dtype=np.float32).reshape(1, 4, 8), (rows, 1, 1))
            self.entries.append(dict(name=name, seconds=[.2], first_seconds=.3))
        self.actual = {k: v.copy() for k, v in self.expected.items()}
        base = dict(**identity, suite_sha256=evaluate.suite_hash(spec), timing_protocol=evaluate.TIMING_PROTOCOL,
                    verified_weight_files=model["files"], checkpoint_receipt=receipt, precision={"requested": "fp32"})
        inputs = {}
        for case in self.cases:
            inputs[case["name"] + "__mel"] = np.zeros((len(case["clips"]), 64, 128), dtype=np.float32)
            inputs[case["name"] + "__attention_mask"] = np.ones((len(case["clips"]), 64), dtype=np.int64)
            inputs[case["name"] + "__mel_lengths"] = np.full(len(case["clips"]), 64, dtype=np.int64)
            inputs[case["name"] + "__audio_seconds"] = np.full(len(case["clips"]), 2.0, dtype=np.float32)
        evaluate.write_json(self.args.input / "suite.json", spec)
        evaluate.write_json(self.args.input / "config.json", model["architecture"])
        evaluate.write_json(self.args.input / "checkpoint_receipt.json", receipt)
        evaluate.write_json(self.args.output / "measurements.json", {**base, "cases": self.entries})
        evaluate.write_json(self.args.oracle / "reference.json",
                            {**base, "cases": [dict(name=c["name"], transcripts=["REF WORDS"] * len(c["clips"]))
                                               for c in self.cases]})
        evaluate.write_json(self.args.oracle / "labels.json", {c["name"]: ["REF WORDS"] * len(c["clips"]) for c in self.cases})
        np.savez(self.args.input / "inputs.npz", **inputs)
        np.savez(self.args.output / "actual.npz", **self.actual)
        np.savez(self.args.oracle / "oracle.npz", **self.expected)
        return spec

    def test_passing_bringup_not_accepted(self):
        result = evaluate.assess(self.args)
        self.assertTrue(result["passed"], result["failures"])
        self.assertFalse(result["accepted"])
        check = result["checks"][0]
        self.assertEqual(check["exact_reference_rows"], 2)
        self.assertEqual(check["audio_seconds"], 4.0)
        self.assertAlmostEqual(check["real_time_factor"], 20.0)

    def test_token_mismatch_fails_even_with_good_encoder(self):
        self.actual["single__tokens"][0, 1] = 99
        np.savez(self.args.output / "actual.npz", **self.actual)
        self.assertFalse(evaluate.assess(self.args)["passed"])

    def test_encoder_nrmse_gate(self):
        self.actual["batch__encoder"][1] += .5
        np.savez(self.args.output / "actual.npz", **self.actual)
        result = evaluate.assess(self.args)
        self.assertFalse(result["passed"])
        self.assertIn("NRMSE", " ".join(result["failures"]))

    def test_revisit_consistency_passes_with_matching_rows(self):
        result = evaluate.assess(self.args)
        check = {c["name"]: c for c in result["checks"]}
        self.assertEqual(check["single"]["exact_reference_rows"], 1)

    def test_identity_tampering_fails(self):
        path = self.args.output / "measurements.json"
        original = json.loads(path.read_text())
        for field, value in (("suite_sha256", "0" * 64), ("model_key", "other"), ("timing_protocol", "x")):
            evaluate.write_json(path, {**original, field: value})
            self.assertFalse(evaluate.assess(self.args)["passed"])
            evaluate.write_json(path, original)

    def test_full_wer_inflation_gate(self):
        self.stage = "full"
        self.cases = [dict(name="w0", clips=["c0"], numerical=True, quality=True),
                      dict(name="w1", clips=["c1"], numerical=True, quality=True)]
        self.strict_fixture()
        # reference transcripts match labels -> reference WER 0; candidate misses one word
        meta = json.loads((self.args.oracle / "reference.json").read_text())
        np.savez(self.args.output / "actual.npz", **self.actual)
        tokenizer = types.SimpleNamespace(batch_decode=lambda rows, **kw: ["REF WRONG"] * len(rows))
        with patch.dict("sys.modules", {"transformers": types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *a, **kw: tokenizer))}):
            result = evaluate.assess(self.args)
        self.assertFalse(result["passed"])
        self.assertTrue(any("WER inflation" in f for f in result["failures"]))
        self.assertEqual(len(result["quality"]), 2)


if __name__ == "__main__":
    unittest.main()
