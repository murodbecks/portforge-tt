"""Trusted evaluator template: suites, FP32 oracle, TT measurement and independent gates.

Copy ports/template/ to ports/<your-model>/ and fill in every TODO. The four
functions below are the whole interface the harness relies on:

  make_suite(corpus, stage, model) -> (spec, labels)
      Build the frozen smoke / bringup / full suites from the corpus.
  reference(args)      runs on the reference host: FP32 oracle outputs + public inputs.
  candidate(args)      runs in the TT sandbox: imports the agent's backend.py, times it.
  assess(args)         runs on the reference host: compares candidate with oracle.

Closest working examples:
  float outputs, one forward pass     ports/esm2 (logits), ports/chronos2 (quantiles)
  token outputs, autoregressive loop  ports/nllb (translation), ports/parakeet (ASR)

Keep the candidate's view minimal: input/ holds only what the backend needs
(never oracle outputs or ground-truth labels). Learned compute must run on TT;
candidate() rejects CPU matmul/attention calls while the backend runs.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time

MODEL = "TODO/hf-repo-id"            # must match models.json
REVISION = "TODO-40-char-commit-sha"
# After building corpus.json, from the harness directory:
#   python -c "import json, cluster; print(cluster.identity(json.load(open('ports/<name>/corpus.json'))))"
CORPUS_SHA256 = "0" * 64
GATES = {"nrmse": .04, "behavior_max_nrmse": .01}   # add task metrics (WER, chrF, ...) as needed
PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TIMING_PROTOCOL = "inputs_to_outputs_with_sync_excluding_load_preprocessing_progress_v1"

import portforge_eval  # noqa: E402
from portforge_eval import (Progress, check_suite, suite_hash, validate_config,  # noqa: E402,F401
                            validate_input_preservation, validate_receipt, verify_weights, write_json)

KEY = "my-model"                     # registry key in models.json and port.json


def model_identity(name=KEY):
    return portforge_eval.model_identity(name)


def suite_model(spec):
    return portforge_eval.suite_model(spec, KEY, GATES)


def validate_artifact(artifact, spec, identity):
    return portforge_eval.validate_artifact(artifact, spec, identity, TIMING_PROTOCOL)


def require_tt_precision(mode, runtime):
    return portforge_eval.require_tt_precision(mode, runtime, PRECISIONS)


def validate_precision(mode, policy, runtime=None):
    return portforge_eval.validate_precision(mode, policy, runtime, PRECISIONS)


def make_suite(corpus, stage, model=KEY):
    """Frozen suites. smoke: 1 case; bringup: ~8 cases covering batching, masking and edge
    inputs; full: the whole held-out corpus plus consistency cases.

    Each case is a dict with a unique "name", its input "clips" (keys into the corpus),
    "numerical": True when it is timed, optional "quality": True for task-metric cases,
    and optional "compare_to"/"compare_rows" for single-row / reversed / revisit
    consistency against another case of the same stage.
    """
    if corpus.get("schema_version") != 1:
        raise ValueError("unsupported corpus schema")
    identity = model_identity(model)
    cases, labels = [], {}

    def add(name, clips, numerical=True, **extra):
        cases.append(dict(name=name, clips=list(clips), numerical=numerical, **extra))
        labels[name] = [corpus["cases"][clip].get("label") for clip in clips]

    # TODO: build the three stages from corpus["selection"], e.g.
    #   if stage == "smoke": add("short", [corpus["selection"]["smoke"]])
    #   elif stage == "bringup": add("batch", [...]); add("single0", [...], compare_to="batch", compare_rows=[0]) ...
    #   else: one case per held-out clip with quality=True, then batch4 / batch4-reverse / revisit.
    raise NotImplementedError("make_suite: define the smoke, bringup and full suites")
    spec = dict(version=2, stage=stage, **identity, corpus_sha256=CORPUS_SHA256, gates=GATES.copy(),  # noqa
                timed_repeats=3 if stage == "full" else 1, cases=cases)
    return check_suite(spec), labels


def reference(args):
    """Reference host: run the model's official implementation in FP32 and write

      <out>/input/   suite.json, config.json, checkpoint_receipt.json, inputs.npz  (public)
      <out>/oracle/  oracle.npz, reference.json, labels.json                        (private)

    Follow ports/esm2/evaluate.py reference(): verify the checkpoint with
    models.verify_checkpoint, record the precision policy, time each case with
    CUDA synchronisation, and disable TF32 for both matmuls and cuDNN
    convolutions so the oracle really is FP32.
    """
    portforge_eval.reference_execution()
    raise NotImplementedError("reference: produce public inputs and FP32 oracle outputs")


def forward_case(backend, case, inputs):
    """TODO: call the agent's backend for one case and return (outputs, supplied_inputs).

    ``inputs`` holds this case's arrays from inputs.npz. Pass copies to the backend,
    return every output array to compare with the oracle (e.g. {"logits": ...}), and
    the copies so mutation can be detected. Validate output shapes here.
    """
    raise NotImplementedError("forward_case: call backend.<your method> and return its arrays")


def candidate(args):
    """TT sandbox: import the agent's backend.py, check its precision declaration, and time
    every case with device synchronisation. This body is generic; only forward_case changes."""
    public, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((public / "suite.json").read_text())
    selected, identity = suite_model(spec)
    config = json.loads((public / "config.json").read_text())
    validate_config(config, selected)
    progress = Progress(out, spec["stage"])
    progress.mark("verify_checkpoint", initializing=True)
    validate_receipt(json.loads((public / "checkpoint_receipt.json").read_text()), identity, selected, spec)
    verified = verify_weights(args.weights, identity["model_key"])
    progress.mark("import_runtime", initializing=True)
    import numpy as np
    import ttnn
    from torch.utils._python_dispatch import TorchDispatchMode
    require_tt_precision(args.precision, ttnn)

    class RejectCPUCompute(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func).startswith(("aten.mm.", "aten.bmm.", "aten.addmm.", "aten.matmul.", "aten.linear.",
                                     "aten.embedding.", "aten.native_layer_norm.", "aten.layer_norm.",
                                     "aten._softmax.", "aten.softmax.", "aten.gelu.", "aten.silu.",
                                     "aten.relu.", "aten.scaled_dot_product", "aten.convolution.")):
                raise RuntimeError("CPU learned compute forbidden: " + str(func))
            return func(*args, **(kwargs or {}))

    calls = [0]
    for name in ("matmul", "linear", "embedding", "layer_norm", "conv1d", "conv2d", "execute_trace"):
        original = getattr(ttnn, name, None)
        if original:
            def counted(*a, _op=original, **kw):
                calls[0] += 1
                return _op(*a, **kw)
            setattr(ttnn, name, counted)
    sys.path.insert(0, os.getcwd())
    backend_module = importlib.import_module("backend")
    options = getattr(backend_module, "DEVICE_OPTIONS", {})
    bounds = {"trace_region_size": (0, 128 * 1024 ** 2), "l1_small_size": (0, 128 * 1024), "num_hw_cqs": (1, 2)}
    if (not isinstance(options, dict) or set(options) - set(bounds)
            or any(type(v) is not int or not bounds[k][0] <= v <= bounds[k][1] for k, v in options.items())):
        raise ValueError("invalid DEVICE_OPTIONS")
    inputs = np.load(public / "inputs.npz", allow_pickle=False)
    arrays, measured = {}, dict(cases=[], device_options=options, **identity, suite_sha256=suite_hash(spec),
                                timing_protocol=TIMING_PROTOCOL, verified_weight_files=verified)
    progress.mark("open_device", initializing=True)
    device = ttnn.open_device(device_id=0, **options)
    try:
        progress.mark("create_backend", initializing=True)
        start = time.perf_counter()
        backend = backend_module.create_backend(args.weights, config, device, precision=args.precision)
        measured["precision"] = validate_precision(args.precision, getattr(backend, "precision_policy", None), ttnn)
        ttnn.synchronize_device(device)
        measured["load_seconds"] = time.perf_counter() - start
        view = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM)
        measured["resident_dram_bytes"] = int(view.num_banks) * int(view.total_bytes_allocated_per_bank)
        for case in spec["cases"]:
            name = case["name"]
            progress.mark("case_begin", name)
            case_inputs = {key.split("__", 1)[1]: inputs[key] for key in inputs.files if key.startswith(name + "__")}
            first, samples, cold = None, [], None
            for repeat in range((spec["timed_repeats"] if case["numerical"] else 1) + 1):
                ttnn.synchronize_device(device)
                progress.mark("forward", name, repeat)
                before, start = calls[0], time.perf_counter()
                with RejectCPUCompute():
                    outputs, supplied = forward_case(backend, case, case_inputs)
                ttnn.synchronize_device(device)
                elapsed = time.perf_counter() - start
                validate_input_preservation(case_inputs, supplied)
                if calls[0] == before:
                    raise ValueError("missing TT compute")
                if first is None:
                    first, cold = {k: np.array(v) for k, v in outputs.items()}, elapsed
                elif any(not np.array_equal(first[k], np.asarray(v)) for k, v in outputs.items()):
                    raise ValueError("TT nondeterminism")
                else:
                    samples.append(elapsed)
            arrays.update({name + "__" + key: value for key, value in first.items()})
            measured["cases"].append(dict(name=name, seconds=samples, first_seconds=cold))
        np.savez_compressed(out / "actual.npz", **arrays)
        measured.update(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        observed_tt_calls=calls[0])
        write_json(out / "measurements.json", measured)
    finally:
        progress.mark("close_device")
        ttnn.close_device(device)
    progress.mark("complete")


def assess(args):
    """Reference host: compare actual.npz with oracle.npz and write verdict.json.

    Per case: shapes, finiteness, max per-row NRMSE <= GATES["nrmse"], exact equality
    for token outputs, compare_to consistency <= GATES["behavior_max_nrmse"], and task
    metrics on quality cases for the full stage. The verdict must carry stage, suite_sha256,
    precision, per-case timing and a list of failures; see ports/esm2/evaluate.py assess().
    """
    raise NotImplementedError("assess: implement the gates for this task")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--stage", choices=("smoke", "bringup", "full"), required=True)
    ref.add_argument("--model", default=KEY)
    for flag in ("corpus", "out"):
        ref.add_argument("--" + flag, required=True)
    for command, flags in (("candidate", ("input", "output", "weights")), ("assess", ("input", "oracle", "output")),
                           ("reference-benchmark", ("input", "oracle", "output"))):
        child = sub.add_parser(command)
        for flag in flags:
            child.add_argument("--" + flag, required=True)
        if command != "assess":
            child.add_argument("--precision", choices=PRECISIONS if command == "candidate" else ("fp32", "bf16"),
                               default="bf16")
        if command == "reference-benchmark":
            child.add_argument("--model", default=KEY)
    args = parser.parse_args()
    result = {"reference": reference, "reference-benchmark": reference, "candidate": candidate,
              "assess": assess}[args.command](args)
    if args.command in ("assess", "reference-benchmark"):
        print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
        sys.exit(0 if result["passed"] else 1)
