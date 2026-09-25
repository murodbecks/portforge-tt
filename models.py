"""Pinned checkpoint identities (models.json); verify bytes before loading model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import re


# A frozen run harness keeps its pack's models.json beside this file; the
# launcher and tests point at a pack's registry with use_registry().
REGISTRY = Path(__file__).with_name("models.json")


def use_registry(path):
    global REGISTRY
    REGISTRY = Path(path)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def model_spec(name, registry=None):
    document = read_json(registry or REGISTRY)
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported model registry schema")
    if name not in document["models"]:
        raise ValueError(f"Unknown model {name!r}; choose {', '.join(document['models'])}")
    spec = document["models"][name]
    if not re.fullmatch(r"[a-f0-9]{40}", spec["revision"]):
        raise ValueError("Model revision must be an immutable commit")
    for filename, identity in spec["files"].items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", filename):
            raise ValueError(f"Invalid checkpoint filename: {filename}")
        if not re.fullmatch(r"[a-f0-9]{64}", identity["sha256"]):
            raise ValueError(f"Invalid SHA-256: {filename}")
        if type(identity["bytes"]) is not int or identity["bytes"] <= 0:
            raise ValueError(f"Invalid file size: {filename}")
    weights = spec["weight_files"]
    index = spec["weight_index"]
    if not weights or len(set(weights)) != len(weights):
        raise ValueError("Weight files must be nonempty and unique")
    required = ["config.json", *weights, *([index] if index else [])]
    if any(filename not in spec["files"] for filename in required):
        raise ValueError("Checkpoint metadata is missing a required file identity")
    formats = {
        "pytorch_bin": (None, 1),
        "pytorch_bin_sharded": ("pytorch_model.bin.index.json", None),
        "safetensors": (None, 1),
    }
    if spec["checkpoint_format"] not in formats:
        raise ValueError("Unsupported checkpoint format")
    expected_index, expected_count = formats[spec["checkpoint_format"]]
    if expected_index and index != expected_index:
        raise ValueError("Checkpoint format does not match weight files/index")
    if not expected_index:
        if index:
            raise ValueError("Single-file checkpoints must not declare a shard index")
        if len(weights) != expected_count:
            raise ValueError("Checkpoint format does not match weight files")
        if spec["checkpoint_format"] == "safetensors" and not all(
                w.endswith(".safetensors") for w in weights):
            raise ValueError("Safetensors checkpoints must use .safetensors weight files")
    return spec


def verify_checkpoint(directory, name, *, registry=None, all_files=False):
    """Hash without deserializing weight files; accept normal HF cache symlinks.

    Default verifies config.json and every weight shard. all_files also checks
    the recorded documentation files (README). A passed receipt proves
    checkpoint identity, not model correctness or support on an accelerator.
    """
    spec = model_spec(name, registry)
    directory = Path(directory)
    filenames = list(spec["files"]) if all_files else ["config.json", *spec["weight_files"]]
    if spec["weight_index"] and spec["weight_index"] not in filenames:
        filenames.append(spec["weight_index"])
    verified = {}
    for filename in filenames:
        path, expected = directory / filename, spec["files"][filename]
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing file or incorrect byte count: {filename}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {filename}")
        verified[filename] = dict(expected)
    config = read_json(directory / "config.json")
    for key, expected in spec["architecture"].items():
        if config.get(key) != expected:
            raise ValueError(f"Architecture mismatch: {key}")
    if spec["weight_index"]:
        weight_map = read_json(directory / spec["weight_index"]).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Shard index must contain a nonempty weight_map")
        if any(not isinstance(value, str) for value in weight_map.values()):
            raise ValueError("Shard index filenames must be strings")
        if set(weight_map.values()) != set(spec["weight_files"]):
            raise ValueError("Shard index does not match the pinned weight files")
    return {"model": name, "repo_id": spec["repo_id"], "revision": spec["revision"],
            "checkpoint_format": spec["checkpoint_format"], "verified_files": verified}


def download_checkpoint(cache, name):
    """Use the installed HF client; restrict downloads to the pinned manifest."""
    from huggingface_hub import snapshot_download
    spec = model_spec(name)
    directory = snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                                  cache_dir=str(cache), allow_patterns=list(spec["files"]), token=False)
    receipt = verify_checkpoint(directory, name, all_files=True)
    return dict(receipt, directory=str(directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="registry key in models.json")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--registry", type=Path, help="a pack's models.json (default: beside this file)")
    parser.add_argument("--all-files", action="store_true")
    parser.add_argument("--download", action="store_true", help="Treat directory as HF cache; download and verify all pinned files")
    args = parser.parse_args()
    if args.registry:
        use_registry(args.registry)
    result = (download_checkpoint(args.directory, args.model) if args.download else
              verify_checkpoint(args.directory, args.model, all_files=args.all_files))
    print(json.dumps(result, indent=2))
