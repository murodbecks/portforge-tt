"""Checkpoint identity tests use tiny fake bytes, never download model weights."""
import support  # noqa: F401  (pack and registry setup)
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import models


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = self.root / "registry.json"
        self.spec = dict(repo_id="test/model", revision="a" * 40,
                         checkpoint_format="safetensors", weight_files=["model.safetensors"],
                         weight_index=None, architecture={"d_model": 4}, files={})
        self.write("config.json", b'{"d_model":4}')
        self.write("model.safetensors", b"fake safetensors bytes")

    def write(self, name, data):
        (self.root / name).write_bytes(data)
        self.spec["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    def save(self):
        self.registry.write_text(json.dumps({"schema_version": 1, "models": {"tiny": self.spec}}))

    def verify(self):
        self.save()
        return models.verify_checkpoint(self.root, "tiny", registry=self.registry)

    def test_real_registry_pins_parakeet_safetensors(self):
        spec = models.model_spec("parakeet")
        self.assertEqual(spec["repo_id"], "nvidia/parakeet-tdt-0.6b-v3")
        self.assertEqual(spec["checkpoint_format"], "safetensors")
        self.assertIsNone(spec["weight_index"])
        self.assertEqual(spec["weight_files"], ["model.safetensors"])
        arch = spec["architecture"]
        self.assertEqual(arch["encoder_config"]["hidden_size"], 1024)
        self.assertEqual(arch["encoder_config"]["num_hidden_layers"], 24)
        self.assertEqual(arch["encoder_config"]["subsampling_factor"], 8)
        self.assertEqual(arch["vocab_size"], 8193)
        self.assertEqual(arch["durations"], [0, 1, 2, 3, 4])
        self.assertEqual(len(spec["files"]["model.safetensors"]["sha256"]), 64)
        for extra in ("tokenizer.json", "processor_config.json"):
            self.assertIn(extra, spec["files"])
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            models.model_spec("parakeet-tdt")

    def test_safetensors_receipt(self):
        self.assertEqual(set(self.verify()["verified_files"]), {"config.json", "model.safetensors"})
        self.assertEqual(self.verify()["checkpoint_format"], "safetensors")

    def test_sharded_bin_registry_still_supported_for_future_models(self):
        self.write("one.bin", b"first weights")
        self.write("two.bin", b"other weights")
        self.write("pytorch_model.bin.index.json", b'{"weight_map":{"encoder":"one.bin","decoder":"two.bin"}}')
        self.spec.update(checkpoint_format="pytorch_bin_sharded", weight_files=["one.bin", "two.bin"],
                         weight_index="pytorch_model.bin.index.json")
        self.assertEqual(len(self.verify()["verified_files"]), 4)

    def test_same_length_corruption_and_missing_weight(self):
        size = (self.root / "model.safetensors").stat().st_size
        (self.root / "model.safetensors").write_bytes(b"X" * size)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.verify()
        (self.root / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "Missing file"):
            self.verify()

    def test_bin_named_safetensors_registry_rejected_and_index_for_single_file(self):
        self.spec.update(weight_files=["model.bin"])
        self.write("model.bin", b"bin weights")
        with self.assertRaisesRegex(ValueError, "Safetensors checkpoints"):
            self.verify()
        self.spec.update(checkpoint_format="pytorch_bin")
        self.verify()
        self.write("index.json", b"{}")
        self.spec["weight_index"] = "index.json"
        with self.assertRaisesRegex(ValueError, "must not declare a shard index"):
            self.verify()

    def test_architecture_mismatch_even_with_matching_file_hash(self):
        self.write("config.json", b'{"d_model":8}')
        with self.assertRaisesRegex(ValueError, "Architecture mismatch"):
            self.verify()

    def test_nested_encoder_config_must_match(self):
        self.write("config.json", b'{"d_model":4,"encoder_config":{"num_hidden_layers":32}}')
        self.spec["architecture"] = {"d_model": 4, "encoder_config": {"num_hidden_layers": 32}}
        self.verify()
        self.spec["architecture"]["encoder_config"]["num_hidden_layers"] = 16
        with self.assertRaisesRegex(ValueError, "encoder_config"):
            self.verify()

    def test_unsafe_filename_unpinned_revision_and_missing_hash(self):
        self.spec["files"]["../outside"] = self.spec["files"]["model.safetensors"]
        with self.assertRaisesRegex(ValueError, "Invalid checkpoint filename"):
            self.verify()
        del self.spec["files"]["../outside"]
        self.spec["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "immutable commit"):
            self.verify()
        self.spec["revision"] = "a" * 40
        del self.spec["files"]["model.safetensors"]
        with self.assertRaisesRegex(ValueError, "missing a required file"):
            self.verify()

    def test_hf_style_cache_symlink_is_verified_by_content(self):
        (self.root / "model.safetensors").rename(self.root / "blob")
        (self.root / "model.safetensors").symlink_to(self.root / "blob")
        self.assertIn("model.safetensors", self.verify()["verified_files"])

    def test_download_is_pinned_anonymous_and_rejects_corrupt_cache(self):
        download = Mock(return_value=str(self.root))
        with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(snapshot_download=download)}), \
             patch.object(models, "model_spec", return_value=self.spec):
            receipt = models.download_checkpoint(self.root / "cache", "tiny")
            self.assertEqual(set(receipt["verified_files"]), {"config.json", "model.safetensors"})
            self.assertEqual(download.call_args.kwargs["revision"], self.spec["revision"])
            self.assertEqual(set(download.call_args.kwargs["allow_patterns"]), set(self.spec["files"]))
            self.assertIs(download.call_args.kwargs["token"], False)
            size = (self.root / "model.safetensors").stat().st_size
            (self.root / "model.safetensors").write_bytes(b"X" * size)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                models.download_checkpoint(self.root / "cache", "tiny")


if __name__ == "__main__":
    unittest.main()
