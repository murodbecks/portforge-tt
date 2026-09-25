"""Trusted real-model workloads, FP32 oracle, TT measurement and independent gates.

Only input/ is exposed to the candidate. Run reference/assess on the configured
reference compute host (CUDA preferred, CPU allowed and recorded); the candidate
imports the agent's backend.py in its current working directory on the TT host.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import statistics
import sys
import time
from models import model_spec, verify_checkpoint

MODEL = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
WEIGHT_SHA256 = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
CORPUS_SHA256 = "f0b08df4fe80385da1d3e9a05e3779708c4e808b2da581aa974fea9505d64ae3"
GATES = {"nrmse": .04, "behavior_max_nrmse": .01, "max_wql_inflation": .02}
PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TIMING_PROTOCOL = "cpu_arrays_to_cpu_arrays_with_sync_excluding_load_preprocessing_progress_v1"
MAX_CONTEXT = 512
MAX_HORIZON = 64
QUANTILES = tuple(model_spec("chronos2")["architecture"]["chronos_config"]["quantiles"])

import portforge_eval
from portforge_eval import (Progress, reference_execution, suite_hash, validate_config, validate_input_preservation,
                            validate_receipt, verify_weights, write_json)

KEY = "chronos2"


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



def series_window(series, context, horizon, window):
    """Deterministic disjoint [context|future] slices; the future tail is ground truth only."""
    start = window * (context + horizon)
    if start + context + horizon > len(series):
        raise ValueError("window exceeds the frozen series")
    return series[start:start + context], series[start + context:start + context + horizon]


def masked_context(values, mask_spec):
    """Apply the frozen missing-value pattern; masked cells keep zero placeholders."""
    mask = [1.0] * len(values)
    context = list(values)
    if mask_spec:
        if "tail" in mask_spec:
            for i in range(len(context) - mask_spec["tail"], len(context)):
                context[i], mask[i] = 0.0, 0.0
        for hole in mask_spec.get("holes", ()):
            context[hole], mask[hole] = 0.0, 0.0
    return context, mask


def make_suite(corpus, stage, model="chronos2"):
    selected, model_fields = model_spec(model), model_identity(model)
    identity = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    if identity != CORPUS_SHA256 or stage not in ("smoke", "bringup", "full"):
        raise ValueError("unrecognized corpus or stage")
    series = corpus["series"]
    cases, labels = [], {}
    def add(name, keys, context, horizon, window=0, mask=None, **extra):
        rows, masks, futures = [], [], []
        for key in keys:
            past, future = series_window(series[key], context, horizon, window)
            values, observed = masked_context(past, mask)
            rows.append(values)
            masks.append(observed)
            futures.append(future)
        cases.append(dict(name=name, series=list(keys), context=context, horizon=horizon,
                          window=window, mask=mask, numerical=True, **extra))
        labels[name] = futures
    if stage == "smoke":
        add("short", ["etth1_ot"], 512, 64)
    elif stage == "bringup":
        add("batch", ["etth1_hufl", "etth1_ot"], 512, 64)
        for name, rows in (("single0", [0]), ("single1", [1]), ("reverse", [1, 0])):
            add(name, [["etth1_hufl", "etth1_ot"][i] for i in rows], 512, 64,
                compare_to="batch", compare_rows=rows)
        add("masked-tail", ["etth1_mull"], 512, 64, mask={"tail": 16})
        add("length-64", ["etth1_lufl"], 64, 64)
        add("length-65", ["etth1_lull"], 65, 64)
        add("constant", ["syn_constant"], 512, 64)
    else:
        for key in [k for k in series if k != "syn_constant"]:
            add(f"{key}-w0", [key], 512, 64, window=0, quality=True)
            add(f"{key}-w1", [key], 128, 16, window=1, quality=True)
            add(f"{key}-w2", [key], 64, 64, window=2, quality=True)
        add("batch4", ["etth1_hufl", "etth1_ot", "syn_sine_mixed", "syn_arcsinh_stress"], 512, 64, quality=True)
        add("masked-holes", ["etth1_mull"], 512, 64, mask={"holes": [3, 17, 251, 500]}, quality=True)
        add("constant", ["syn_constant"], 512, 64, quality=True)
        for name, rows in (("single0", [0]), ("single1", [1]), ("reverse", [3, 2, 1, 0])):
            add("batch4-" + name, [["etth1_hufl", "etth1_ot", "syn_sine_mixed", "syn_arcsinh_stress"][i] for i in rows],
                512, 64, compare_to="batch4", compare_rows=rows)
        add("revisit", ["etth1_hufl", "etth1_ot", "syn_sine_mixed", "syn_arcsinh_stress"], 512, 64,
            compare_to="batch4", compare_rows=[0, 1, 2, 3])
        add("masked-tail", ["etth1_mull"], 512, 64, mask={"tail": 16})
    return dict(version=2, stage=stage, **model_fields,
                weight_sha256=selected["files"][selected["weight_files"][0]]["sha256"] if not selected["weight_index"] else None,
                corpus_sha256=CORPUS_SHA256, max_context=MAX_CONTEXT, gates=GATES.copy(),
                quantiles=list(QUANTILES), timed_repeats=3 if stage == "full" else 1, cases=cases), labels


def quantile_nrmse(actual, expected):
    """Max per-row NRMSE over the (horizon, quantiles) forecast block."""
    import numpy as np
    if actual.shape != expected.shape or actual.dtype.kind != "f" or expected.dtype.kind != "f":
        raise ValueError("quantile shape/dtype check")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite quantile values")
    axes = tuple(range(1, actual.ndim))
    error = np.sqrt(np.mean((actual - expected) ** 2, axis=axes)) / np.maximum(
        1e-12, np.sqrt(np.mean(expected ** 2, axis=axes)))
    if not np.isfinite(error).all():
        raise ValueError("nonfinite NRMSE")
    return float(np.max(error))


def pinball_loss(quantiles, future, levels):
    """Mean pinball loss over steps and quantiles, scaled by the mean absolute target."""
    import numpy as np
    tau = np.asarray(levels, dtype=np.float64).reshape((1, -1))
    diff = future[:, :, None] - quantiles
    loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)
    scale = float(np.mean(np.abs(future)))
    return float(np.mean(loss) / max(scale, 1e-12))


def load_pipeline(directory, dtype, device):
    """Load through the public chronos-forecasting package; verify the pinned revision."""
    from chronos import Chronos2Pipeline
    try:
        return Chronos2Pipeline.from_pretrained(str(directory), dtype=dtype, device_map=device)
    except TypeError:
        return Chronos2Pipeline.from_pretrained(str(directory), torch_dtype=dtype, device_map=device)


def extract_quantiles(predictions, rows, horizon):
    """predict() -> list of (n_variates, n_quantiles, horizon) tensors; univariate only here."""
    import torch
    if len(predictions) != rows:
        raise ValueError("pipeline returned one tensor per univariate series expected")
    stacked = []
    for item in predictions:
        tensor = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != horizon:
            raise ValueError(f"unexpected forecast tensor shape {tuple(tensor.shape)}")
        stacked.append(tensor[0])  # [Q, H]
    return torch.stack(stacked).cpu().float().numpy()  # [B, Q, H]


def reference(args):
    execution = reference_execution()
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        torch.manual_seed(1729)
    benchmark = getattr(args, "command", "reference") == "reference-benchmark"
    precision = args.precision if benchmark else "fp32"
    if precision not in ("fp32", "bf16"):
        raise ValueError("the chronos-forecasting pipeline exposes float32/bfloat16 references only")
    dtype = {"fp32": "float32", "bf16": "bfloat16"}[precision]
    root = Path(args.output if benchmark else args.out)
    root.mkdir(parents=True, exist_ok=True)
    public, private = root / "input", root / "oracle"
    if benchmark:
        public, private = Path(args.input), Path(args.oracle)
        oracle_meta = json.loads((private / "reference.json").read_text())
        if oracle_meta.get("precision", {}).get("requested", "fp32") != "fp32":
            raise ValueError("precision comparisons require the fixed FP32 oracle")
        spec = json.loads((public / "suite.json").read_text())
        if spec["stage"] not in ("smoke", "bringup"):
            raise ValueError("precision benchmark is diagnostic: use smoke or bringup inputs")
        prepared = np.load(public / "inputs.npz", allow_pickle=False)
    else:
        public.mkdir(exist_ok=True)
        private.mkdir(exist_ok=True)
        corpus = json.loads(Path(args.corpus).read_text())
        spec, labels = make_suite(corpus, args.stage, getattr(args, "model", "chronos2"))
        if precision != "fp32":
            raise ValueError("the immutable correctness oracle is FP32")
    selected, identity = suite_model(spec)
    if benchmark and getattr(args, "model", None) not in (None, identity["model_key"]):
        raise ValueError("benchmark model differs from prepared inputs")
    if benchmark:
        validate_artifact(oracle_meta, spec, identity)
        validate_config(json.loads((public / "config.json").read_text()), selected)
        validate_receipt(oracle_meta.get("checkpoint_receipt", {}), identity, selected, spec)
    progress = Progress(root, spec["stage"])
    progress.mark("load_model", initializing=True)
    torch.set_num_threads(12)
    torch.manual_seed(1729)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        device = "cuda"
        if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("this CUDA device lacks native BF16 support")
    else:
        device = "cpu"
    started = time.perf_counter()
    from huggingface_hub import hf_hub_download
    for filename in selected["files"]:
        progress.mark("download_checkpoint", filename, initializing=True)
        path = Path(hf_hub_download(selected["repo_id"], filename, revision=selected["revision"], token=False))
    directory = path.parent
    progress.mark("verify_checkpoint", initializing=True)
    receipt = verify_checkpoint(directory, identity["model_key"], all_files=True)
    receipt["model_manifest_sha256"] = identity["model_manifest_sha256"]
    progress.mark("load_model", initializing=True)
    pipeline = load_pipeline(directory, dtype, device)
    if str(device) not in str(getattr(pipeline, "model", pipeline).device) and device == "cuda":
        raise RuntimeError("reference pipeline did not land on CUDA")
    progress.mark("validate_config", initializing=True)
    validate_config(json.loads((directory / "config.json").read_text()), selected)
    if not benchmark:
        (public / "config.json").write_text((directory / "config.json").read_text())
    meta = dict(load_seconds=time.perf_counter() - started, device=device,
                gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                hostname=socket.gethostname(), execution=execution, job_id=os.getenv("SLURM_JOB_ID"),
                torch_version=torch.__version__, **identity, checkpoint_receipt=receipt,
                verified_weight_files=receipt["verified_files"], timing_protocol=TIMING_PROTOCOL,
                parameter_dtypes=[dtype],
                precision=validate_precision(precision, dict(mode=precision, weights=precision,
                    activations=precision, accumulation=f"PyTorch {device} operator-dependent; no autocast",
                    exceptions=[])), cases=[])
    inputs, expected = {}, {}
    with torch.inference_mode():
        for case in spec["cases"]:
            name = case["name"]
            progress.mark("case_begin", name)
            if benchmark:
                past = torch.from_numpy(prepared[name + "__past_values"].copy())
                mask = torch.from_numpy(prepared[name + "__past_observed_mask"].copy())
            else:
                values, mask_rows = [], []
                corpus_series = corpus["series"]
                for key in case["series"]:
                    history, _future = series_window(corpus_series[key], case["context"], case["horizon"], case["window"])
                    context, observed = masked_context(history, case.get("mask"))
                    values.append(context)
                    mask_rows.append(observed)
                past = torch.tensor(values, dtype=torch.float32)
                mask = torch.tensor(mask_rows, dtype=torch.float32)
            inputs.update({name + "__past_values": past.numpy(), name + "__past_observed_mask": mask.numpy()})
            context = past.clone()
            context[mask == 0] = float("nan")  # the package expects NaN for missing observations
            batch = context.unsqueeze(1)  # [B, 1 variate, T]
            progress.mark("predict", name)
            first, samples, cold = None, [], None
            for repeat in range((spec["timed_repeats"] if case["numerical"] else 1) + 1):
                progress.mark("predict_call", name, repeat)
                start = time.perf_counter()
                predictions = pipeline.predict(batch, prediction_length=case["horizon"],
                                               batch_size=batch.shape[0], limit_prediction_length=True)
                elapsed = time.perf_counter() - start
                progress.mark("predict_synchronize", name, repeat)
                start_sync = time.perf_counter()
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - start_sync
                quantiles = extract_quantiles(predictions, batch.shape[0], case["horizon"])  # [B, Q, H]
                if first is None:
                    first, cold = quantiles, elapsed
                else:
                    if not np.array_equal(first, quantiles):
                        raise ValueError("reference nondeterminism")
                    samples.append(elapsed)
            expected[name + "__quantiles"] = first.transpose(0, 2, 1)  # [B, H, Q]
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            else:
                peak_allocated = peak_reserved = None
            meta["cases"].append(dict(name=name, seconds=samples, first_seconds=cold,
                                      peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved))
            progress.mark("case_complete", name)
    progress.mark("write_outputs")
    meta["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    meta["suite_sha256"] = receipt["suite_sha256"] = suite_hash(spec)
    if benchmark:
        np.savez_compressed(root / "actual.npz", **expected)
        write_json(root / "measurements.json", meta)
        result = assess(args)
        write_json(root / "verdict.json", result)
        progress.mark("complete")
        return result
    np.savez_compressed(public / "inputs.npz", **inputs)
    np.savez_compressed(private / "oracle.npz", **expected)
    write_json(public / "suite.json", spec)
    write_json(public / "checkpoint_receipt.json", receipt)
    write_json(private / "labels.json", labels)
    write_json(private / "reference.json", meta)
    progress.mark("complete")


def candidate(args):
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
    import torch  # noqa: F401  (weight conversion helpers on the TT host)
    import ttnn
    from torch.utils._python_dispatch import TorchDispatchMode
    require_tt_precision(args.precision, ttnn)
    class RejectCPUCompute(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func).startswith(("aten.mm.", "aten.bmm.", "aten.addmm.", "aten.matmul.", "aten.linear.",
                "aten.embedding.", "aten.native_layer_norm.", "aten.layer_norm.", "aten._softmax.",
                "aten.softmax.", "aten.gelu.", "aten.relu.", "aten.scaled_dot_product", "aten.sinh.")):
                raise RuntimeError("CPU learned compute forbidden: " + str(func))
            return func(*args, **(kwargs or {}))
    calls = [0]
    for name in ("matmul", "linear", "embedding", "layer_norm", "execute_trace"):
        original = getattr(ttnn, name, None)
        if original:
            def counted(*a, _name=name, _op=original, **kw):
                calls[0] += 1
                return _op(*a, **kw)
            setattr(ttnn, name, counted)
    sys.path.insert(0, os.getcwd())
    backend_module = importlib.import_module("backend")
    options = getattr(backend_module, "DEVICE_OPTIONS", {})
    bounds = {"trace_region_size": (0, 128 * 1024 ** 2), "l1_small_size": (0, 128 * 1024), "num_hw_cqs": (1, 2)}
    if not isinstance(options, dict) or set(options) - set(bounds) or any(type(v) is not int or not bounds[k][0] <= v <= bounds[k][1] for k, v in options.items()):
        raise ValueError("invalid DEVICE_OPTIONS")
    inputs, arrays, measured = np.load(public / "inputs.npz", allow_pickle=False), {}, dict(cases=[], device_options=options,
        **identity, suite_sha256=suite_hash(spec), timing_protocol=TIMING_PROTOCOL, verified_weight_files=verified)
    progress.mark("open_device", initializing=True)
    device = ttnn.open_device(device_id=0, **options)
    try:
        progress.mark("create_backend", initializing=True)
        start = time.perf_counter()
        backend = backend_module.create_backend(args.weights, config, device,
                                                precision=args.precision)
        measured["precision"] = validate_precision(args.precision, getattr(backend, "precision_policy", None), ttnn)
        progress.mark("load_synchronize", initializing=True)
        ttnn.synchronize_device(device)
        measured["load_seconds"] = time.perf_counter() - start
        view = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM)
        measured["resident_dram_bytes"] = int(view.num_banks) * int(view.total_bytes_allocated_per_bank)
        measured["trace_reserved_bytes"] = int(view.num_banks) * options.get("trace_region_size", 0)
        for case in spec["cases"]:
            name = case["name"]
            progress.mark("case_begin", name)
            past, mask = inputs[name + "__past_values"], inputs[name + "__past_observed_mask"]
            first, samples = None, []
            for repeat in range((spec["timed_repeats"] if case["numerical"] else 1) + 1):
                progress.mark("predict_pre_synchronize", name, repeat)
                ttnn.synchronize_device(device)
                progress.mark("predict", name, repeat)
                before, start = calls[0], time.perf_counter()
                supplied = dict(past_values=past.copy(), past_observed_mask=mask.copy())
                with RejectCPUCompute():
                    result = backend.forecast(supplied["past_values"], supplied["past_observed_mask"],
                                              case["horizon"])
                elapsed = time.perf_counter() - start
                progress.mark("predict_synchronize", name, repeat)
                start = time.perf_counter()
                ttnn.synchronize_device(device)
                elapsed += time.perf_counter() - start
                validate_input_preservation(dict(past_values=past, past_observed_mask=mask), supplied)
                quantiles = np.asarray(result["quantiles"], dtype=np.float32)
                if quantiles.shape != (past.shape[0], case["horizon"], len(spec["quantiles"])):
                    raise ValueError("forecast quantiles must be [B,H,Q]")
                if calls[0] == before:
                    raise ValueError("missing TT compute")
                if first is None:
                    first, cold = quantiles.copy(), elapsed
                else:
                    if not np.array_equal(first, quantiles):
                        raise ValueError("TT nondeterminism")
                    samples.append(elapsed)
            arrays[name + "__quantiles"] = first
            measured["cases"].append(dict(name=name, seconds=samples, first_seconds=cold))
            progress.mark("case_complete", name)
        progress.mark("write_outputs")
        np.savez_compressed(out / "actual.npz", **arrays)
        measured.update(host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        observed_tt_calls=calls[0], memory_scope="resident allocator snapshot, not peak")
        write_json(out / "measurements.json", measured)
    finally:
        progress.mark("close_device")
        ttnn.close_device(device)
    progress.mark("complete")


def assess(args):
    import numpy as np
    public, private, out = Path(args.input), Path(args.oracle), Path(args.output)
    spec = json.loads((public / "suite.json").read_text())
    selected, identity = suite_model(spec)
    strict = spec.get("version", 1) >= 2
    actual, expected = np.load(out / "actual.npz", allow_pickle=False), np.load(private / "oracle.npz", allow_pickle=False)
    measured = json.loads((out / "measurements.json").read_text())
    entries = measured["cases"]
    measurements = {c["name"]: c for c in entries}
    failures, checks = [], []
    if strict:
        try:
            validate_config(json.loads((public / "config.json").read_text()), selected)
            validate_receipt(json.loads((public / "checkpoint_receipt.json").read_text()), identity, selected, spec)
            validate_artifact(measured, spec, identity)
            needed = [*selected["weight_files"], *([selected["weight_index"]] if selected["weight_index"] else [])]
            verified = measured.get("verified_weight_files")
            if not isinstance(verified, dict) or any(verified.get(k) != selected["files"][k] for k in needed):
                raise ValueError("candidate weight verification identity mismatch")
            reference_meta = json.loads((private / "reference.json").read_text())
            validate_artifact(reference_meta, spec, identity)
            validate_receipt(reference_meta.get("checkpoint_receipt", {}), identity, selected, spec)
            if reference_meta.get("precision", {}).get("requested") != "fp32":
                raise ValueError("correctness oracle must be FP32")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append("identity: " + str(exc))
    keys = {c["name"] + "__quantiles" for c in spec["cases"]}
    if set(actual.files) != keys or set(expected.files) != keys:
        failures.append("missing/unexpected arrays")
    if len(entries) != len(spec["cases"]) or set(measurements) != {c["name"] for c in spec["cases"]}:
        failures.append("missing/duplicate timings")
    for case in spec["cases"]:
        name, check = case["name"], {"name": case["name"]}
        try:
            quantiles = actual[name + "__quantiles"]
            shape = (len(case["series"]), case["horizon"], len(spec["quantiles"]))
            if quantiles.shape != shape or quantiles.dtype.kind != "f":
                raise ValueError("invalid quantiles shape/dtype")
            oracle = expected[name + "__quantiles"].astype(np.float64)
            error = quantile_nrmse(quantiles.astype(np.float64), oracle)
            check["rows"] = len(case["series"])
            check["quantile_max_row_nrmse"] = error
            if error > GATES["nrmse"]:
                raise ValueError("quantile NRMSE exceeds 0.04")
            if "compare_to" in case:
                base = actual[case["compare_to"] + "__quantiles"].astype(np.float64)
                for row, source in enumerate(case["compare_rows"]):
                    per_row = quantile_nrmse(quantiles[row][None], base[source][None])
                    if per_row > GATES["behavior_max_nrmse"]:
                        raise ValueError("batch/order/revisit semantics changed for row " + str(row))
            samples = measurements[name]["seconds"]
            if not samples or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError("invalid timing samples")
            if strict and len(samples) != (spec["timed_repeats"] if case["numerical"] else 1):
                raise ValueError("timing repeat count disagrees with prepared suite")
            cold = measurements[name].get("first_seconds")
            if strict and (type(cold) not in (int, float) or not math.isfinite(cold) or cold <= 0):
                raise ValueError("invalid first-call timing")
            check.update(p50_seconds=statistics.median(samples), timing_samples=len(samples), seconds=samples,
                         first_seconds=cold, forecast_points=int(quantiles.shape[0] * quantiles.shape[1]),
                         forecasts_per_second=quantiles.shape[0] / statistics.median(samples))
            crossing = int(np.sum(np.diff(np.sort(quantiles, axis=-1), axis=-1) < 0))
            check["quantile_crossings"] = crossing
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            failures.append(name + ": " + str(exc))
        checks.append(check)
    quality = {}
    if spec["stage"] == "full" and not failures:
        try:
            labels = json.loads((private / "labels.json").read_text())
        except (OSError, ValueError) as exc:
            failures.append("labels: " + str(exc))
            labels = {}
        for case in spec["cases"]:
            if not case.get("quality"):
                continue
            name = case["name"]
            try:
                future = np.asarray(labels[name], dtype=np.float64)
                candidate_loss = pinball_loss(actual[name + "__quantiles"].astype(np.float64), future, spec["quantiles"])
                reference_loss = pinball_loss(expected[name + "__quantiles"].astype(np.float64), future, spec["quantiles"])
                if not math.isfinite(candidate_loss) or not math.isfinite(reference_loss):
                    raise ValueError("nonfinite task loss")
            except (KeyError, ValueError, IndexError, TypeError) as exc:
                failures.append(name + ": " + str(exc))
                continue
            inflation = candidate_loss / max(reference_loss, 1e-12) - 1
            quality[name] = dict(wql=candidate_loss, reference_wql=reference_loss, inflation=inflation)
            if inflation > GATES["max_wql_inflation"]:
                failures.append(name + f": weighted quantile-loss inflation exceeds {GATES['max_wql_inflation']:.0%}")
    nrmse = [c.get("quantile_max_row_nrmse") for c in checks if c.get("quantile_max_row_nrmse") is not None]
    timings = [c["p50_seconds"] for c in checks if "p50_seconds" in c]
    verdict = dict(stage=spec["stage"], **identity, suite_sha256=suite_hash(spec), timing_protocol=TIMING_PROTOCOL,
                   passed=not failures, accepted=not failures and spec["stage"] == "full",
                   failures=failures, case_count=len(checks), max_nrmse=max(nrmse, default=None),
                   timing={"median_case_p50_seconds": statistics.median(timings) if timings else None,
                           "measured_cases": len(timings)},
                   precision=measured.get("precision"), checks=checks, quality=quality, production_certified=False)
    write_json(out / "verdict.json", verdict)
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--stage", choices=("smoke", "bringup", "full"), required=True)
    ref.add_argument("--model", default="chronos2", choices=("chronos2",))
    for flag in ("corpus", "out"):
        ref.add_argument("--" + flag, required=True)
    for command, flags in (("candidate", ("input", "output", "weights")), ("assess", ("input", "oracle", "output")),
                           ("reference-benchmark", ("input", "oracle", "output"))):
        child = sub.add_parser(command)
        for flag in flags:
            child.add_argument("--" + flag, required=True)
        if command != "assess":
            child.add_argument("--precision", choices=PRECISIONS if command == "candidate" else ("fp32", "bf16"), default="bf16")
        if command == "reference-benchmark":
            child.add_argument("--model", choices=("chronos2",))
    args = parser.parse_args()
    result = {"reference": reference, "reference-benchmark": reference, "candidate": candidate, "assess": assess}[args.command](args)
    if args.command in ("assess", "reference-benchmark"):
        print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
        sys.exit(0 if result["passed"] else 1)
