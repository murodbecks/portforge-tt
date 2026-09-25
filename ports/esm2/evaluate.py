"""Trusted protein workloads, FP32 oracle, TT measurement and independent gates.

Only input/ is exposed to the candidate: already-masked token ids + attention
mask + the frozen masked-position list. Never the raw sequences' recovery
answers. The oracle runs the public transformers EsmForMaskedLM in FP32 on the
configured reference host (CUDA preferred, CPU recorded).
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

MODEL = "facebook/esm2_t33_650M_UR50D"
REVISION = "08e4846e537177426273712802403f7ba8261b6c"
WEIGHT_SHA256 = "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"
CORPUS_SHA256 = "369c475fa78f1aad39963eee3fc43344c62be766793ed529e12fc1fdb4326a9e"
GATES = {"nrmse": .04, "behavior_max_nrmse": .01}
PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TIMING_PROTOCOL = "token_ids_to_logits_with_sync_excluding_load_tokenization_progress_v1"
MAX_RESIDUES = 1024
MASK_STRIDE, MASK_OFFSET = 16, 5  # deterministic masked positions: 5, 21, 37, ...

import portforge_eval
from portforge_eval import (Progress, reference_execution, suite_hash, validate_config, validate_input_preservation,
                            validate_receipt, verify_weights, write_json)

KEY = "esm2"


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



def masked_positions(length):
    return [p for p in range(MASK_OFFSET, length, MASK_STRIDE)]


def make_suite(corpus, stage, model="esm2"):
    selected, model_fields = model_spec(model), model_identity(model)
    identity = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    if identity != CORPUS_SHA256 or stage not in ("smoke", "bringup", "full"):
        raise ValueError("unrecognized corpus or stage")
    entries = {entry["name"]: entry for entry in corpus["cases"]}
    cases = []
    def add(name, keys, **extra):
        cases.append(dict(name=name, sequences=list(keys), numerical=True, **extra))
    if stage == "smoke":
        add("short", [corpus["selection"]["smoke"]])
    elif stage == "bringup":
        pick = corpus["selection"]["bringup"]
        add("batch", [pick["batch0"], pick["batch1"]])
        for name, rows in (("single0", [0]), ("single1", [1]), ("reverse", [1, 0])):
            add(name, [pick["batch" + str(i)] for i in rows], compare_to="batch", compare_rows=rows)
        add("long", [pick["long"]])
        add("short", [pick["short"]])
        add("unknown-residues", [pick["unknown"]])
        add("single-residue", ["syn-single"])
    else:
        for key in corpus["selection"]["full"]:
            add(key, [key], quality=True)
        pick = corpus["selection"]["bringup"]
        batch4 = [pick["batch0"], pick["batch1"], corpus["selection"]["full"][0], corpus["selection"]["full"][-1]]
        add("batch4", batch4, quality=True)
        add("batch4-reverse", batch4[::-1], compare_to="batch4", compare_rows=[3, 2, 1, 0])
        # batch4 rows 0-1 hold the same two sequences; the full stage has no "batch" case.
        add("revisit", batch4[:2], compare_to="batch4", compare_rows=[0, 1])
        add("single-residue", ["syn-single"])
    return dict(version=2, stage=stage, **model_fields,
                weight_sha256=selected["files"][selected["weight_files"][0]]["sha256"] if not selected["weight_index"] else None,
                corpus_sha256=CORPUS_SHA256, max_residues=MAX_RESIDUES, gates=GATES.copy(),
                mask_stride=MASK_STRIDE, mask_offset=MASK_OFFSET, vocab_size=selected["architecture"]["vocab_size"],
                timed_repeats=3 if stage == "full" else 1, cases=cases), {}


def max_row_nrmse(actual, expected):
    import numpy as np
    if actual.shape != expected.shape or actual.dtype.kind != "f" or expected.dtype.kind != "f":
        raise ValueError("shape/dtype check")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite values")
    axes = tuple(range(1, actual.ndim))
    error = np.sqrt(np.mean((actual - expected) ** 2, axis=axes)) / np.maximum(
        1e-12, np.sqrt(np.mean(expected ** 2, axis=axes)))
    if not np.isfinite(error).all():
        raise ValueError("nonfinite NRMSE")
    return float(np.max(error))


def reference(args):
    execution = reference_execution()
    import numpy as np
    import torch
    torch.manual_seed(1729)
    benchmark = getattr(args, "command", "reference") == "reference-benchmark"
    precision = args.precision if benchmark else "fp32"
    if not benchmark and precision != "fp32":
        raise ValueError("the immutable correctness oracle is FP32")
    if precision not in ("fp32", "bf16"):
        raise ValueError("the transformers reference exposes float32/bfloat16 only")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[precision]
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
        corpus = None
    else:
        public.mkdir(exist_ok=True)
        private.mkdir(exist_ok=True)
        corpus = json.loads(Path(args.corpus).read_text())
        spec, _ = make_suite(corpus, args.stage, getattr(args, "model", "esm2"))
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
    from transformers import EsmForMaskedLM, EsmTokenizerFast
    model = EsmForMaskedLM.from_pretrained(str(directory), dtype=dtype, local_files_only=True,
                                           trust_remote_code=False, weights_only=True,
                                           use_safetensors=True).eval().to(device)
    tokenizer = EsmTokenizerFast.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    progress.mark("validate_config", initializing=True)
    validate_config(json.loads((directory / "config.json").read_text()), selected)
    if not benchmark:
        (public / "config.json").write_text((directory / "config.json").read_text())
    meta = dict(load_seconds=time.perf_counter() - started, device=device,
                gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                hostname=socket.gethostname(), execution=execution, job_id=os.getenv("SLURM_JOB_ID"),
                torch_version=torch.__version__, **identity, checkpoint_receipt=receipt,
                verified_weight_files=receipt["verified_files"], timing_protocol=TIMING_PROTOCOL,
                parameter_dtypes=[str(dtype).split(".")[-1]],
                precision=validate_precision(precision, dict(mode=precision, weights=precision,
                    activations=precision, accumulation=f"PyTorch {device} operator-dependent; no autocast",
                    exceptions=[])), cases=[])
    mask_id = selected["architecture"]["mask_token_id"]
    inputs, expected = {}, {}
    entries = {entry["name"]: entry for entry in corpus["cases"]} if corpus else None
    with torch.inference_mode():
        for case in spec["cases"]:
            name = case["name"]
            progress.mark("case_begin", name)
            if benchmark:
                ids = torch.from_numpy(prepared[name + "__input_ids"].copy()).long()
                mask = torch.from_numpy(prepared[name + "__attention_mask"].copy()).long()
            else:
                seqs = [entries[key]["sequence"] for key in case["sequences"]]
                if any(len(s) > MAX_RESIDUES for s in seqs):
                    raise ValueError("sequence exceeds the frozen envelope")
                enc = tokenizer(seqs, padding=True, return_tensors="pt")
                ids, mask = enc.input_ids.clone(), enc.attention_mask
                case["masked_positions"] = [masked_positions(len(seq)) for seq in seqs]
                for row, positions in enumerate(case["masked_positions"]):
                    for p in positions:  # residue p sits at token index p+1 (cls at 0)
                        ids[row, p + 1] = mask_id
                inputs.update({name + "__input_ids": ids.numpy().astype(np.int64),
                               name + "__attention_mask": mask.numpy().astype(np.int64),
                               name + "__residues": np.array([len(s) for s in seqs], dtype=np.int64)})
            progress.mark("forward", name)
            output = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                           output_hidden_states=True)
            progress.mark("forward_readback", name)
            expected[name + "__logits"] = output.logits.float().cpu().numpy()
            expected[name + "__hidden"] = output.hidden_states[-1].float().cpu().numpy()
            first, samples, cold = None, [], None
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for repeat in range(spec["timed_repeats"] + 1):
                progress.mark("forward_timed", name, repeat)
                start = time.perf_counter()
                output = model(input_ids=ids.to(device), attention_mask=mask.to(device))
                elapsed = time.perf_counter() - start
                progress.mark("forward_synchronize", name, repeat)
                start_sync = time.perf_counter()
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - start_sync
                if first is None:
                    first, cold = output.logits.float().cpu().numpy(), elapsed
                else:
                    if not np.array_equal(first, output.logits.float().cpu().numpy()):
                        raise ValueError("reference nondeterminism")
                    samples.append(elapsed)
            meta["cases"].append(dict(name=name, seconds=samples, first_seconds=cold,
                                      peak_allocated_bytes=(torch.cuda.max_memory_allocated() if device == "cuda" else None),
                                      peak_reserved_bytes=(torch.cuda.max_memory_reserved() if device == "cuda" else None)))
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
    write_json(private / "labels.json", {})
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
    import torch  # noqa: F401  (CPU weight conversion helpers on the TT host)
    import ttnn
    from torch.utils._python_dispatch import TorchDispatchMode
    require_tt_precision(args.precision, ttnn)
    class RejectCPUCompute(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func).startswith(("aten.mm.", "aten.bmm.", "aten.addmm.", "aten.matmul.", "aten.linear.",
                "aten.embedding.", "aten.native_layer_norm.", "aten.layer_norm.", "aten._softmax.",
                "aten.softmax.", "aten.gelu.", "aten.silu.", "aten.relu.", "aten.scaled_dot_product")):
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
        backend = backend_module.create_backend(args.weights, config, device, precision=args.precision)
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
            ids, mask = inputs[name + "__input_ids"], inputs[name + "__attention_mask"]
            first, samples = None, []
            for repeat in range((spec["timed_repeats"] if case["numerical"] else 1) + 1):
                progress.mark("forward_pre_synchronize", name, repeat)
                ttnn.synchronize_device(device)
                progress.mark("forward", name, repeat)
                before, start = calls[0], time.perf_counter()
                supplied = dict(input_ids=ids.copy(), attention_mask=mask.copy())
                with RejectCPUCompute():
                    result = backend.embed(supplied["input_ids"], supplied["attention_mask"])
                elapsed = time.perf_counter() - start
                progress.mark("forward_synchronize", name, repeat)
                start = time.perf_counter()
                ttnn.synchronize_device(device)
                elapsed += time.perf_counter() - start
                validate_input_preservation(dict(input_ids=ids, attention_mask=mask), supplied)
                logits = np.asarray(result["logits"], dtype=np.float32)
                hidden = np.asarray(result["hidden"], dtype=np.float32)
                if logits.shape != ids.shape + (selected["architecture"]["vocab_size"],):
                    raise ValueError("logits must be [B,L,vocab]")
                if hidden.shape != ids.shape + (selected["architecture"]["hidden_size"],):
                    raise ValueError("hidden must be [B,L,hidden]")
                if calls[0] == before:
                    raise ValueError("missing TT compute")
                if first is None:
                    first, cold = (logits.copy(), hidden.copy()), elapsed
                else:
                    if not (np.array_equal(first[0], logits) and np.array_equal(first[1], hidden)):
                        raise ValueError("TT nondeterminism")
                    samples.append(elapsed)
            arrays[name + "__logits"], arrays[name + "__hidden"] = first
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
    inputs = np.load(public / "inputs.npz", allow_pickle=False) if strict else None
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
    keys = {c["name"] + "__" + k for c in spec["cases"] for k in ("logits", "hidden")}
    if set(actual.files) != keys or set(expected.files) != keys:
        failures.append("missing/unexpected arrays")
    if len(entries) != len(spec["cases"]) or set(measurements) != {c["name"] for c in spec["cases"]}:
        failures.append("missing/duplicate timings")
    vocab = selected["architecture"]["vocab_size"]
    for case in spec["cases"]:
        name, check = case["name"], {"name": case["name"]}
        try:
            ids = inputs[name + "__input_ids"] if strict else None
            masked = case.get("masked_positions", []) if strict else None
            for key in ("logits", "hidden"):
                a = actual[name + "__" + key].astype(np.float64)
                b = expected[name + "__" + key].astype(np.float64)
                error = max_row_nrmse(a, b)
                check[key + "_max_row_nrmse"] = error
                if error > GATES["nrmse"]:
                    raise ValueError(key + " NRMSE exceeds 0.04")
            check["rows"] = actual[name + "__logits"].shape[0]
            if strict:
                predicted, oracle = [], []
                for row, positions in enumerate(masked):
                    predicted.append([int(actual[name + "__logits"][row, p + 1].argmax()) for p in positions])
                    oracle.append([int(expected[name + "__logits"][row, p + 1].argmax()) for p in positions])
                check["exact_masked_rows"] = sum(p == o for p, o in zip(predicted, oracle))
                if check["exact_masked_rows"] != check["rows"]:
                    raise ValueError("masked-position argmax differs from FP32 reference")
            if "compare_to" in case and strict:
                base = actual[case["compare_to"] + "__hidden"].astype(np.float64)
                mine = actual[name + "__hidden"].astype(np.float64)
                for row, source in enumerate(case["compare_rows"]):
                    if max_row_nrmse(mine[row][None], base[source][None]) > GATES["behavior_max_nrmse"]:
                        raise ValueError("batch/order/revisit semantics changed for row " + str(row))
            samples = measurements[name]["seconds"]
            if not samples or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError("invalid timing samples")
            if strict and len(samples) != spec["timed_repeats"]:
                raise ValueError("timing repeat count disagrees with prepared suite")
            cold = measurements[name].get("first_seconds")
            if strict and (type(cold) not in (int, float) or not math.isfinite(cold) or cold <= 0):
                raise ValueError("invalid first-call timing")
            residues = int(inputs[name + "__residues"].sum()) if strict else 0
            check.update(p50_seconds=statistics.median(samples), timing_samples=len(samples), seconds=samples,
                         first_seconds=cold, residues=residues,
                         residues_per_second=residues / statistics.median(samples) if residues else None)
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            failures.append(name + ": " + str(exc))
        checks.append(check)
    nrmse = [v for check in checks for k, v in check.items() if k.endswith("_nrmse")]
    timings = [c["p50_seconds"] for c in checks if "p50_seconds" in c]
    verdict = dict(stage=spec["stage"], **identity, suite_sha256=suite_hash(spec), timing_protocol=TIMING_PROTOCOL,
                   passed=not failures, accepted=not failures and spec["stage"] == "full",
                   failures=failures, case_count=len(checks), max_nrmse=max(nrmse, default=None),
                   timing={"median_case_p50_seconds": statistics.median(timings) if timings else None,
                           "measured_cases": len(timings)},
                   precision=measured.get("precision"), checks=checks, quality={}, production_certified=False)
    write_json(out / "verdict.json", verdict)
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--stage", choices=("smoke", "bringup", "full"), required=True)
    ref.add_argument("--model", default="esm2", choices=("esm2",))
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
            child.add_argument("--model", choices=("esm2",))
    args = parser.parse_args()
    result = {"reference": reference, "reference-benchmark": reference, "candidate": candidate, "assess": assess}[args.command](args)
    if args.command in ("assess", "reference-benchmark"):
        print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
        sys.exit(0 if result["passed"] else 1)
