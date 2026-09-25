"""Trusted real-audio workloads, FP32 oracle, TT measurement and independent gates.

Only input/ is exposed to the candidate (mel features + lengths; never raw audio,
never transcripts). Run reference/assess on the configured reference host (CUDA
preferred, CPU allowed and recorded); the candidate imports the agent's
backend.py in its current working directory on the TT host. Deterministic mel
preprocessing (STFT/log-mel) is declared policy and runs on the host CPU, like
tokenization did for translation; learned compute must run on TT/CUDA.
"""
import argparse
import base64
import hashlib
import importlib
import io
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

MODEL = "nvidia/parakeet-tdt-0.6b-v3"
REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
WEIGHT_SHA256 = "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11"
CORPUS_SHA256 = '62fb9c517719931999ccd72b9c097e889e843aaf8345567f62f16e95e8248d02'
GATES = {"nrmse": .04, "max_wer_inflation": .02, "behavior_max_nrmse": .01}
PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TIMING_PROTOCOL = "mel_features_to_tokens_with_sync_excluding_load_preprocessing_progress_v1"
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 32

import portforge_eval
from portforge_eval import (Progress, reference_execution, suite_hash, validate_config, validate_input_preservation,
                            validate_receipt, verify_weights, write_json)

KEY = "parakeet"


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



def load_audio(entry):
    """Decode one frozen corpus clip to mono float32 samples at 16 kHz."""
    import numpy as np
    import soundfile as sf
    data, rate = sf.read(io.BytesIO(base64.b64decode(entry["audio_b64"])), dtype="float32")
    if data.ndim != 1:
        data = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise ValueError("frozen corpus clips are pinned at 16 kHz; got " + str(rate))
    if len(data) > MAX_AUDIO_SECONDS * SAMPLE_RATE:
        raise ValueError("clip exceeds the frozen envelope")
    return np.asarray(data, dtype=np.float32)


def make_suite(corpus, stage, model="parakeet"):
    selected, model_fields = model_spec(model), model_identity(model)
    identity = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    if identity != CORPUS_SHA256 or stage not in ("smoke", "bringup", "full"):
        raise ValueError("unrecognized corpus or stage")
    entries = {entry["name"]: entry for entry in corpus["cases"]}
    cases, labels = [], {}
    def add(name, keys, numerical=True, **extra):
        rows = [entries[key] for key in keys]
        cases.append(dict(name=name, clips=list(keys), numerical=numerical, **extra))
        labels[name] = [row["transcript"] for row in rows]
    if stage == "smoke":
        add("short", [corpus["selection"]["smoke"]])
    elif stage == "bringup":
        pick = corpus["selection"]["bringup"]
        add("batch", [pick["batch0"], pick["batch1"]])
        for name, rows in (("single0", [0]), ("single1", [1]), ("reverse", [1, 0])):
            add(name, [pick["batch" + str(i)] for i in rows], compare_to="batch", compare_rows=rows)
        add("long", [pick["long"]])
        add("short", [pick["short"]])
        add("silence", [pick["silence"]])
        add("tone", ["syn-tone"])
    else:
        for key in corpus["selection"]["full"]:
            add(key, [key], quality=True)
        pick = corpus["selection"]["bringup"]
        batch4 = [pick["batch0"], pick["batch1"], corpus["selection"]["full"][0], corpus["selection"]["full"][-1]]
        add("batch4", batch4, quality=True)
        add("batch4-reverse", batch4[::-1], compare_to="batch4", compare_rows=[3, 2, 1, 0])
        # batch4 rows 0-1 hold the same two clips; the full stage has no "batch" case.
        add("revisit", batch4[:2], compare_to="batch4", compare_rows=[0, 1])
        add("silence", [pick["silence"]])
        add("long", [pick["long"]])
    return dict(version=2, stage=stage, **model_fields,
                weight_sha256=selected["files"][selected["weight_files"][0]]["sha256"] if not selected["weight_index"] else None,
                corpus_sha256=CORPUS_SHA256, max_audio_seconds=MAX_AUDIO_SECONDS, gates=GATES.copy(),
                timed_repeats=3 if stage == "full" else 1, cases=cases), labels


def canonical_tokens(row, blank, pad, vocab):
    """Strip padding after the first blank-free greedy sequence; validate ids."""
    row = [int(x) for x in row]
    if any(x < 0 or x >= vocab for x in row):
        raise ValueError("token outside pinned vocabulary")
    end = len(row)
    while end > 0 and row[end - 1] == pad:
        end -= 1
    return row[:end]


def word_error_rate(hypothesis, reference_text):
    """Word-level Levenshtein distance over whitespace tokens, normalized by reference length."""
    hyp, ref = hypothesis.split(), reference_text.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(ref) + 1))
    for i, word in enumerate(hyp, 1):
        current = [i]
        for j, ref_word in enumerate(ref, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (word != ref_word)))
        previous = current
    return previous[-1] / len(ref)


def quantile_free_nrmse(actual, expected):
    """Max per-row NRMSE over the encoder time dimension."""
    import numpy as np
    if actual.shape != expected.shape or actual.dtype.kind != "f" or expected.dtype.kind != "f":
        raise ValueError("encoder shape/dtype check")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite encoder values")
    axes = tuple(range(1, actual.ndim))
    error = np.sqrt(np.mean((actual - expected) ** 2, axis=axes)) / np.maximum(
        1e-12, np.sqrt(np.mean(expected ** 2, axis=axes)))
    if not np.isfinite(error).all():
        raise ValueError("nonfinite NRMSE")
    return float(np.max(error))


def load_parakeet(directory, dtype, device):
    """Load through the public transformers implementation of ParakeetForTDT."""
    import torch
    import transformers
    cls = getattr(transformers, "AutoModelForTDT", None) or getattr(transformers, "ParakeetForTDT", None)
    if cls is None:
        raise RuntimeError("installed transformers lacks Parakeet TDT support (need >= 5.6)")
    return cls.from_pretrained(str(directory), dtype=dtype, local_files_only=True,
                               trust_remote_code=False, weights_only=True, use_safetensors=True).eval().to(device)


def encode_reference(model, mel, frame_mask, device):
    """Encoder last hidden states through the public implementation, with shape fallbacks."""
    import torch
    inner = model.model if hasattr(model, "model") else model
    with torch.inference_mode():
        dtype = next(model.parameters()).dtype  # bf16 study: features must match the weights
        enc = inner.encoder(input_features=mel.to(device=device, dtype=dtype),
                            attention_mask=frame_mask.to(device))
        states = enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc
        if states is None:
            raise RuntimeError("reference encoder produced no hidden states; verify the installed transformers API")
        return states


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
        spec, labels = make_suite(corpus, args.stage, getattr(args, "model", "parakeet"))
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
        # cuDNN convolutions default to TF32 on Ampere+; the FP32 oracle must not use it.
        torch.backends.cudnn.allow_tf32 = False
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
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    model = load_parakeet(directory, dtype, device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
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
    inputs, expected = {}, {}
    with torch.inference_mode():
        for case in spec["cases"]:
            name = case["name"]
            progress.mark("case_begin", name)
            if benchmark:
                mel = torch.from_numpy(prepared[name + "__mel"].copy())
                mel_lengths = torch.from_numpy(prepared[name + "__mel_lengths"].copy()).long()
                frame_mask = torch.from_numpy(prepared[name + "__attention_mask"].copy()).long()
            else:
                entries = {entry["name"]: entry for entry in corpus["cases"]}
                waveforms = [load_audio(entries[key]) for key in case["clips"]]
                if any(len(w) == 0 for w in waveforms):
                    raise ValueError("empty frozen clip")
                features = processor(waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
                # Parakeet processor: input_features [B, T, 128 mel] (time-major),
                # attention_mask [B, T] with row sums = true frame counts.
                mel = features.input_features
                if not torch.is_tensor(mel):
                    mel = torch.stack([torch.as_tensor(m) for m in mel])
                mel = mel.to(torch.float32)
                frame_mask = features.attention_mask.long()
                mel_lengths = frame_mask.sum(dim=-1)
                inputs.update({name + "__mel": mel.numpy().astype(np.float32),
                               name + "__mel_lengths": mel_lengths.numpy().astype(np.int64),
                               name + "__attention_mask": frame_mask.numpy().astype(np.int64),
                               name + "__audio_seconds": np.array([len(w) / SAMPLE_RATE for w in waveforms],
                                                                  dtype=np.float32)})
            if case["numerical"]:
                progress.mark("encode", name)
                encoded = encode_reference(model, mel, frame_mask, device)
                progress.mark("encode_readback", name)
                expected[name + "__encoder"] = encoded.float().cpu().numpy()
            first, samples, cold = None, [], None
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for repeat in range((spec["timed_repeats"] if case["numerical"] else 1) + 1):
                progress.mark("generate", name, repeat)
                start = time.perf_counter()
                generated = model.generate(input_features=mel.to(device=device, dtype=next(model.parameters()).dtype),
                                           attention_mask=frame_mask.to(device),
                                           return_dict_in_generate=True)
                elapsed = time.perf_counter() - start
                progress.mark("generate_synchronize", name, repeat)
                start_sync = time.perf_counter()
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - start_sync
                tokens = generated.sequences.cpu().numpy()
                if first is None:
                    first, cold = tokens, elapsed
                else:
                    if not np.array_equal(first, tokens):
                        raise ValueError("reference nondeterminism")
                    samples.append(elapsed)
            expected[name + "__tokens"] = first
            blank, pad = selected["architecture"]["blank_token_id"], selected["architecture"]["pad_token_id"]
            vocab = selected["architecture"]["vocab_size"]
            canonical = [canonical_tokens(row, blank, pad, vocab) for row in first]
            transcripts = tokenizer.batch_decode([torch.tensor(row) for row in canonical], skip_special_tokens=True)
            meta["cases"].append(dict(name=name, seconds=samples, first_seconds=cold, transcripts=transcripts,
                                      peak_allocated_bytes=(torch.cuda.max_memory_allocated() if device == "cuda" else None),
                                      peak_reserved_bytes=(torch.cuda.max_memory_reserved() if device == "cuda" else None)))
            if not benchmark:
                inputs[name + "__tokens_reference_shape"] = np.array(first.shape, dtype=np.int64)
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
    import torch  # noqa: F401  (CPU weight conversion helpers on the TT host)
    import ttnn
    from torch.utils._python_dispatch import TorchDispatchMode
    require_tt_precision(args.precision, ttnn)
    class RejectCPUCompute(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func).startswith(("aten.mm.", "aten.bmm.", "aten.addmm.", "aten.matmul.", "aten.linear.",
                "aten.embedding.", "aten.native_layer_norm.", "aten.layer_norm.", "aten._softmax.",
                "aten.softmax.", "aten.gelu.", "aten.silu.", "aten.scaled_dot_product", "aten.conv2d.",
                "aten.conv1d.", "aten._convolution.")):
                raise RuntimeError("CPU learned compute forbidden: " + str(func))
            return func(*args, **(kwargs or {}))
    calls = [0]
    for name in ("matmul", "linear", "embedding", "layer_norm", "execute_trace", "conv1d", "conv2d"):
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
            mel, mel_lengths = inputs[name + "__mel"], inputs[name + "__mel_lengths"]
            first, samples = None, []
            if case["numerical"]:
                progress.mark("encode", name)
                originals = dict(mel=mel, mel_lengths=mel_lengths)
                supplied = {key: value.copy() for key, value in originals.items()}
                with RejectCPUCompute():
                    result = backend.encode(supplied["mel"], supplied["mel_lengths"])
                progress.mark("encode_synchronize", name)
                ttnn.synchronize_device(device)
                validate_input_preservation(originals, supplied)
                encoder = np.asarray(result["encoder"], dtype=np.float32)
                expected_frames = -(-mel.shape[1] // selected["architecture"]["encoder_config"]["subsampling_factor"])
                if encoder.shape != (mel.shape[0], expected_frames, selected["architecture"]["encoder_config"]["hidden_size"]):
                    raise ValueError("encoder output must be [B, ceil(T/8), hidden]")
                arrays[name + "__encoder"] = encoder
            for repeat in range((spec["timed_repeats"] if case["numerical"] else 1) + 1):
                progress.mark("transcribe_pre_synchronize", name, repeat)
                ttnn.synchronize_device(device)
                progress.mark("transcribe", name, repeat)
                before, start = calls[0], time.perf_counter()
                supplied = dict(mel=mel.copy(), mel_lengths=mel_lengths.copy())
                with RejectCPUCompute():
                    tokens = np.asarray(backend.transcribe(supplied["mel"], supplied["mel_lengths"])["tokens"])
                elapsed = time.perf_counter() - start
                progress.mark("transcribe_synchronize", name, repeat)
                start = time.perf_counter()
                ttnn.synchronize_device(device)
                elapsed += time.perf_counter() - start
                validate_input_preservation(dict(mel=mel, mel_lengths=mel_lengths), supplied)
                if tokens.dtype.kind not in "iu" or tokens.ndim != 2:
                    raise ValueError("transcribe must return integer token ids [B,L]")
                if calls[0] == before:
                    raise ValueError("missing TT compute")
                if first is None:
                    first, cold = tokens.copy(), elapsed
                else:
                    if not np.array_equal(first, tokens):
                        raise ValueError("TT nondeterminism")
                    samples.append(elapsed)
            arrays[name + "__tokens"] = first
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
    public_inputs = np.load(public / "inputs.npz", allow_pickle=False) if strict else {}
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
    keys = {c["name"] + "__" + k for c in spec["cases"] for k in (("tokens", "encoder") if c["numerical"] else ("tokens",))}
    if set(actual.files) != keys or set(expected.files) != keys:
        failures.append("missing/unexpected arrays")
    if len(entries) != len(spec["cases"]) or set(measurements) != {c["name"] for c in spec["cases"]}:
        failures.append("missing/duplicate timings")
    blank, pad, vocab = (selected["architecture"][k] for k in ("blank_token_id", "pad_token_id", "vocab_size"))
    for case in spec["cases"]:
        name, check = case["name"], {"name": case["name"]}
        try:
            tokens = actual[name + "__tokens"]
            if tokens.ndim != 2 or tokens.shape[0] != len(case["clips"]) or tokens.dtype.kind not in "iu":
                raise ValueError("invalid tokens shape/dtype")
            canonical = [canonical_tokens(row, blank, pad, vocab) for row in tokens]
            oracle = [canonical_tokens(row, blank, pad, vocab) for row in expected[name + "__tokens"]]
            check["rows"] = len(case["clips"])
            check["exact_reference_rows"] = sum(a == b for a, b in zip(canonical, oracle))
            if case["numerical"]:
                a = actual[name + "__encoder"].astype(np.float64)
                b = expected[name + "__encoder"].astype(np.float64)
                error = quantile_free_nrmse(a, b)
                check["encoder_max_row_nrmse"] = error
                if error > GATES["nrmse"]:
                    raise ValueError("encoder NRMSE exceeds 0.04")
            if check["exact_reference_rows"] != check["rows"]:
                raise ValueError("greedy tokens differ from FP32 reference")
            if "compare_to" in case:
                base = [canonical_tokens(row, blank, pad, vocab)
                        for row in actual[case["compare_to"] + "__tokens"]]
                for row, source in enumerate(case["compare_rows"]):
                    if canonical[row] != base[source]:
                        raise ValueError("batch/order/revisit semantics changed for row " + str(row))
            samples = measurements[name]["seconds"]
            if not samples or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError("invalid timing samples")
            if strict and len(samples) != (spec["timed_repeats"] if case["numerical"] else 1):
                raise ValueError("timing repeat count disagrees with prepared suite")
            cold = measurements[name].get("first_seconds")
            if strict and (type(cold) not in (int, float) or not math.isfinite(cold) or cold <= 0):
                raise ValueError("invalid first-call timing")
            audio_seconds = float(public_inputs[name + "__audio_seconds"].sum()) if strict else 0.0
            check.update(p50_seconds=statistics.median(samples), timing_samples=len(samples), seconds=samples,
                         first_seconds=cold, audio_seconds=audio_seconds,
                         real_time_factor=audio_seconds / statistics.median(samples) if audio_seconds else None)
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            failures.append(name + ": " + str(exc))
        checks.append(check)
    quality = {}
    if spec["stage"] == "full" and not failures:
        labels = json.loads((private / "labels.json").read_text())
        reference_meta = json.loads((private / "reference.json").read_text())
        reference_transcripts = {c["name"]: c["transcripts"] for c in reference_meta["cases"]}
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(selected["repo_id"], revision=selected["revision"],
                                                       trust_remote_code=False, local_files_only=True)
        except (OSError, ValueError, ImportError) as exc:
            failures.append("labels: tokenizer unavailable on assessor (" + str(exc)[:120] + ")")
            tokenizer = None
        if tokenizer is not None:
            for case in spec["cases"]:
                if not case.get("quality"):
                    continue
                name = case["name"]
                try:
                    rows = [canonical_tokens(r, blank, pad, vocab) for r in actual[name + "__tokens"]]
                    hyp = tokenizer.batch_decode(rows, skip_special_tokens=True)
                except (ValueError, TypeError) as exc:
                    failures.append(name + ": " + str(exc))
                    continue
                human = labels[name]
                oracle_wer = max(word_error_rate(o, h) for o, h in zip(reference_transcripts[name], human))
                candidate_wer = max(word_error_rate(y, h) for y, h in zip(hyp, human))
                if not math.isfinite(candidate_wer) or not math.isfinite(oracle_wer):
                    failures.append(name + ": nonfinite WER")
                    continue
                inflation = candidate_wer - oracle_wer
                quality[name] = dict(wer=candidate_wer, reference_wer=oracle_wer, wer_inflation=inflation)
                if inflation > GATES["max_wer_inflation"]:
                    failures.append(name + f": WER inflation exceeds {GATES['max_wer_inflation']:.0%}")
    nrmse = [c.get("encoder_max_row_nrmse") for c in checks if c.get("encoder_max_row_nrmse") is not None]
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
    ref.add_argument("--model", default="parakeet", choices=("parakeet",))
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
            child.add_argument("--model", choices=("parakeet",))
    args = parser.parse_args()
    result = {"reference": reference, "reference-benchmark": reference, "candidate": candidate, "assess": assess}[args.command](args)
    if args.command in ("assess", "reference-benchmark"):
        print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
        sys.exit(0 if result["passed"] else 1)
