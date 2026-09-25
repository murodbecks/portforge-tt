"""Model-independent parts of a PortForge trusted evaluator.

Every port's ``evaluate.py`` builds its own suites, reference outputs and gates,
but checkpoint identity, receipts, precision declarations, progress reporting
and input-integrity checks are the same for any model. They live here so a new
port only writes the model-specific parts (see ports/template/evaluate.py).

This module runs on the reference host, inside the TT sandbox, and in the
controller, so it depends only on the standard library and ``models.py``.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from models import model_spec, verify_checkpoint

# TT storage formats an evaluator may accept. BFP8_B is block floating point.
TT_PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TT_DTYPES = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16", "bfp8_b": "bfloat8_b"}
POLICY_KEYS = {"mode", "weights", "activations", "accumulation", "exceptions"}
SUITE_VERSION = 2


def suite_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def model_identity(name):
    model = model_spec(name)
    digest = hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(model_key=name, model=model["repo_id"], revision=model["revision"], model_manifest_sha256=digest)


def suite_model(spec, default_key, gates):
    """Check a prepared suite against the pinned registry and frozen gates."""
    name = spec.get("model_key", default_key)
    identity = model_identity(name)
    if spec.get("version") != SUITE_VERSION or any(spec.get(k) != v for k, v in identity.items()):
        raise ValueError("suite model identity does not match the pinned registry")
    if spec.get("gates") != gates:
        raise ValueError("suite changed the frozen gates")
    return model_spec(name), identity


def check_suite(spec):
    """Every consistency case must compare against a case the same suite contains.

    Single-row, reversed-batch and revisit cases compare their outputs against rows
    of another case. Naming a case the stage does not build makes the gate
    impossible to pass, so reject such suites when they are built.
    """
    names = {case["name"] for case in spec["cases"]}
    for case in spec["cases"]:
        if "compare_to" not in case:
            continue
        if case["compare_to"] not in names:
            raise ValueError(f"{spec['stage']} case {case['name']} compares to missing case {case['compare_to']}")
        rows = case.get("compare_rows")
        if (not isinstance(rows, list) or not rows or len(set(rows)) != len(rows)
                or any(type(row) is not int or row < 0 for row in rows)):
            raise ValueError(f"{spec['stage']} case {case['name']} needs distinct nonnegative compare_rows")
    return spec


def validate_config(config, model):
    for key, value in model["architecture"].items():
        if config.get(key) != value:
            raise ValueError("configuration does not match pinned model architecture: " + key)


def validate_receipt(receipt, identity, model, spec=None):
    required = ["config.json", *model["weight_files"], *([model["weight_index"]] if model["weight_index"] else [])]
    if not isinstance(receipt, dict) or not isinstance(receipt.get("verified_files"), dict):
        raise ValueError("malformed checkpoint verification receipt")
    if (receipt.get("model") != identity["model_key"] or receipt.get("repo_id") != identity["model"]
            or receipt.get("revision") != identity["revision"]
            or receipt.get("checkpoint_format") != model["checkpoint_format"]
            or receipt.get("model_manifest_sha256") != identity["model_manifest_sha256"]
            or any(receipt.get("verified_files", {}).get(k) != model["files"][k] for k in required)):
        raise ValueError("checkpoint verification receipt does not match pinned model")
    if spec is not None and receipt.get("suite_sha256") != suite_hash(spec):
        raise ValueError("checkpoint receipt belongs to a different prepared suite")


def validate_artifact(artifact, spec, identity, timing_protocol):
    if (not isinstance(artifact, dict) or any(artifact.get(k) != v for k, v in identity.items())
            or artifact.get("suite_sha256") != suite_hash(spec)
            or artifact.get("timing_protocol") != timing_protocol):
        raise ValueError("artifact model/suite/timing identity mismatch")


def verify_weights(path, name):
    """Single-file checkpoint or an official checkpoint directory; hash bytes, never execute."""
    path, model = Path(path), model_spec(name)
    if path.is_dir():
        return verify_checkpoint(path, name)["verified_files"]
    if model["weight_index"]:
        raise ValueError("sharded model requires a checkpoint directory")
    filename = model["weight_files"][0]
    expected, checksum = model["files"][filename], hashlib.sha256()
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        raise ValueError("candidate weight size mismatch")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != expected["sha256"]:
        raise ValueError("candidate weight SHA-256 mismatch")
    return {filename: dict(expected)}


def reference_execution():
    """The reference must run on an explicitly configured compute host, never a login node."""
    mode = os.getenv("PORTFORGE_REFERENCE_EXECUTION", "slurm")
    if mode not in ("slurm", "direct") or socket.gethostname().lower().startswith(("login", "lo-")):
        raise RuntimeError("reference requires an explicitly configured compute host")
    if mode == "slurm" and not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Slurm reference requires a compute allocation")
    return mode


def require_tt_precision(mode, runtime, precisions=TT_PRECISIONS):
    if mode not in precisions:
        raise ValueError("unsupported precision; TT BFP8 is not IEEE FP8")
    dtype = TT_DTYPES[mode]
    if runtime is not None and getattr(runtime, dtype, None) is None:
        raise ValueError(f"installed TTNN does not expose native {dtype}; no silent precision substitution")


def validate_precision(mode, policy, runtime=None, precisions=TT_PRECISIONS):
    """Validate the backend's declaration; this is not a proof of every tensor's dtype."""
    require_tt_precision(mode, runtime, precisions)
    if not isinstance(policy, dict) or set(policy) != POLICY_KEYS:
        raise ValueError("backend must declare precision_policy fields mode/weights/activations/accumulation/exceptions")
    if policy["mode"] != mode or policy["weights"] != mode or policy["activations"] not in precisions:
        raise ValueError("precision declaration disagrees with requested mode")
    require_tt_precision(policy["activations"], runtime, precisions)
    if not isinstance(policy["accumulation"], str) or not policy["accumulation"].strip():
        raise ValueError("declare accumulation policy explicitly")
    if not isinstance(policy["exceptions"], list) or any(not isinstance(x, str) or not x.strip() for x in policy["exceptions"]):
        raise ValueError("precision exceptions must be explicit nonempty descriptions")
    if policy["activations"] != mode and not policy["exceptions"]:
        raise ValueError("mixed activation precision requires an explanation")
    return dict(requested=mode, effective=dict(policy, exceptions=list(policy["exceptions"])), declaration_only=True,
                format_note="block floating point; not IEEE FP8" if mode == "bfp8_b" else mode)


def validate_input_preservation(expected, observed):
    """Reject caller-array mutation, including changes to shape or dtype."""
    import numpy as np
    if expected.keys() != observed.keys():
        raise ValueError("backend input set changed")
    for name, original in expected.items():
        value = observed[name]
        if (not isinstance(value, np.ndarray) or value.dtype != original.dtype
                or not np.array_equal(value, original, equal_nan=True)):
            raise ValueError("backend mutated caller input: " + name)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    os.replace(temporary, path)


class Progress:
    """Heartbeat file the runner uses to tell a slow phase from a hang."""

    def __init__(self, directory, stage):
        self.path, self.sequence, self.stage = Path(directory) / "progress.json", 0, stage

    def mark(self, phase, case=None, repeat=None, initializing=False):
        self.sequence += 1
        event = dict(version=1, sequence=self.sequence, updated_unix_seconds=time.time(),
                     phase=phase, case=case, repeat=repeat,
                     timeout_seconds=600 if initializing else (300 if self.stage == "full" else 180))
        write_json(self.path, event)
        print(json.dumps({"progress": event}), flush=True)
        return event
