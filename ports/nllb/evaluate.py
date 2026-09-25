"""Trusted real-model workloads, CUDA oracle, TT measurement and independent gates.

Only input/ is exposed to the candidate. Run reference/assess on GPU compute nodes;
candidate imports the agent's backend.py in its current working directory.
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

MODEL = "facebook/nllb-200-distilled-600M"
REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
WEIGHT_SHA256 = "c266c2cfd19758b6d09c1fc31ecdf1e485509035f6b51dfe84f1ada83eefcc42"
CORPUS_SHA256 = "c9b03ca2525db184b6feaf341a52383ec1d9d74311682462c29ba036e74327d1"
LANGUAGES = ("fra_Latn", "uzn_Latn", "arb_Arab", "zho_Hans")
GATES = {"nrmse": .04, "max_chrf_loss": .5, "behavior_exact": True}
PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")
TIMING_PROTOCOL = "cpu_tokens_to_cpu_tokens_with_sync_excluding_load_tokenization_progress_v1"

import portforge_eval
from portforge_eval import (Progress, reference_execution, suite_hash, validate_config, validate_input_preservation,
                            validate_receipt, verify_weights, write_json)

KEY = "600m"


def model_identity(name=KEY):
    return portforge_eval.model_identity(name)


def suite_model(spec):
    """Version-2 suites use the shared check; version 1 is the archived 600M-only format."""
    if spec.get("version") == 1:
        name = spec.get("model_key", KEY)
        identity = model_identity(name)
        if name != KEY or any(k in spec and spec[k] != identity[k] for k in ("model", "revision")):
            raise ValueError("legacy suites support only the pinned 600m model")
        return portforge_eval.model_spec(name), identity
    return portforge_eval.suite_model(spec, KEY, GATES)


def validate_artifact(artifact, spec, identity):
    return portforge_eval.validate_artifact(artifact, spec, identity, TIMING_PROTOCOL)


def require_tt_precision(mode, runtime):
    return portforge_eval.require_tt_precision(mode, runtime, PRECISIONS)


def validate_precision(mode, policy, runtime=None):
    return portforge_eval.validate_precision(mode, policy, runtime, PRECISIONS)



def make_suite(corpus, stage, model="600m"):
    selected, model_fields = model_spec(model), model_identity(model)
    identity = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    if identity != CORPUS_SHA256 or stage not in ("smoke", "bringup", "full"):
        raise ValueError("unrecognized corpus or stage")
    cases, labels = [], {}
    def add(name, texts, src="eng_Latn", tgt="fra_Latn", cap=8, **extra):
        cases.append(dict(name=name, texts=texts, src=src, tgt=tgt,
                          max_new_tokens=cap, numerical=True, **extra))
    if stage == "smoke":
        add("short", ["Hello world."], cap=4)
    elif stage == "bringup":
        add("batch", ["Hello world.", "Good morning, everyone."])
        for name, rows in (("single0", [0]), ("single1", [1]), ("reverse", [1, 0]), ("padding", [0, 1])):
            add(name, [cases[0]["texts"][i] for i in rows], compare_to="batch", compare_rows=rows,
                extra_padding=32 if name == "padding" else 0)
        for length in (32, 33):
            add(f"length-{length}", ["The students are testing translation. " * 100], token_length=length)
        add("unicode-empty", ["", "Hello 👋! Café — €50."])
    else:
        for src, tgt in [("eng_Latn", l) for l in LANGUAGES] + [(l, "eng_Latn") for l in LANGUAGES]:
            for offset in range(0, 1012, 4):
                name = f"{src}-{tgt}-{offset}"
                add(name, corpus[src][offset:offset + 4], src, tgt, cap=64, quality=True)
                cases[-1]["numerical"] = offset == 0
                labels[name] = corpus[tgt][offset:offset + 4]
            first = next(c for c in cases if c["name"] == f"{src}-{tgt}-0")
            for tag, rows in (("single0", [0]), ("single1", [1]), ("single2", [2]),
                              ("single3", [3]), ("pair", [0, 1]), ("triple", [0, 1, 2]),
                              ("reverse", [3, 2, 1, 0]), ("padding", [0, 1, 2, 3])):
                add(first["name"] + "-" + tag, [first["texts"][i] for i in rows], src, tgt,
                    cap=64, compare_to=first["name"], compare_rows=rows,
                    extra_padding=32 if tag == "padding" else 0)
        for length in (2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256):
            add(f"length-{length}", ["The students are testing translation. " * 100], token_length=length)
        add("unicode-empty", ["", " \n\t", "Hello 👋! Café — €50.", "<script>hello</script>"], cap=32)
        for cap in (2, 3, 31, 32, 33, 64):
            add(f"cap-{cap}", ["Hello world."], cap=cap)
        for first in list(cases):
            if first.get("quality") and first["numerical"]:
                add(first["name"] + "-revisit", first["texts"], first["src"], first["tgt"], cap=64,
                    compare_to=first["name"], compare_rows=[0, 1, 2, 3])
    return dict(version=2, stage=stage, **model_fields,
                weight_sha256=selected["files"][selected["weight_files"][0]]["sha256"] if not selected["weight_index"] else None,
                corpus_sha256=CORPUS_SHA256, max_source_tokens=256, gates=GATES.copy(),
                timed_repeats=3 if stage == "full" else 1, cases=cases), labels


def canonical_tokens(row, target, cap, config=None):
    config = config or model_spec("600m")["architecture"]
    start, eos, pad, vocab = (config[k] for k in ("decoder_start_token_id", "eos_token_id", "pad_token_id", "vocab_size"))
    row = [int(x) for x in row]
    if len(row) < 2 or len(row) > cap + 1 or row[:2] != [start, target] or any(x < 0 or x >= vocab for x in row):
        raise ValueError("invalid token range/start/target/length")
    end = row.index(eos, 2) if eos in row[2:] else len(row)
    if pad in row[2:end] or any(x != pad for x in row[end + 1:]) or (end == len(row) and len(row) != cap + 1):
        raise ValueError("invalid EOS/padding/termination")
    return row[:end + 1]


def reference(args):
    execution = reference_execution()
    import numpy as np
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU is not a reference compute substitute")
    benchmark = getattr(args, "command", "reference") == "reference-benchmark"
    precision = args.precision if benchmark else "fp32"
    if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("this CUDA device lacks native BF16 support")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    root = Path(args.output if benchmark else args.out)
    root.mkdir(parents=True, exist_ok=True)
    public, private = root / "input", root / "oracle"
    if benchmark:
        public, private = Path(args.input), Path(args.oracle)
        oracle_meta = json.loads((private / "reference.json").read_text())
        if oracle_meta.get("precision", {}).get("requested", "fp32") != "fp32":
            raise ValueError("precision comparisons require the fixed FP32 oracle")
        spec, labels = json.loads((public / "suite.json").read_text()), {}
        if spec["stage"] not in ("smoke", "bringup"):
            raise ValueError("precision benchmark is diagnostic: use smoke or bringup inputs")
        prepared = np.load(public / "inputs.npz", allow_pickle=False)
    else:
        public.mkdir(exist_ok=True)
        private.mkdir(exist_ok=True)
        spec, labels = make_suite(json.loads(Path(args.corpus).read_text()), args.stage, getattr(args, "model", "600m"))
    selected, identity = suite_model(spec)
    if benchmark and getattr(args, "model", None) not in (None, identity["model_key"]):
        raise ValueError("benchmark model differs from prepared inputs")
    if benchmark:
        validate_artifact(oracle_meta, spec, identity)
        if spec.get("version") == 2:
            validate_config(json.loads((public / "config.json").read_text()), selected)
            validate_receipt(oracle_meta.get("checkpoint_receipt", {}), identity, selected, spec)
    progress = Progress(root, spec["stage"])
    progress.mark("load_model", initializing=True)
    torch.set_num_threads(12)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
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
    model = AutoModelForSeq2SeqLM.from_pretrained(str(directory), dtype=dtype, local_files_only=True,
        attn_implementation="eager", trust_remote_code=False, weights_only=True, use_safetensors=False).eval().cuda()
    progress.mark("load_tokenizer", initializing=True)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False, local_files_only=True)
    architecture = selected["architecture"]
    validate_config(model.config.to_dict(), selected)
    if not benchmark:
        model.config.to_json_file(public / "config.json")
    meta = dict(load_seconds=time.perf_counter() - started, gpu=torch.cuda.get_device_name(0),
                hostname=socket.gethostname(), execution=execution, job_id=os.getenv("SLURM_JOB_ID"),
                torch_version=torch.__version__, **identity, checkpoint_receipt=receipt,
                verified_weight_files=receipt["verified_files"], timing_protocol=TIMING_PROTOCOL,
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                precision=validate_precision(precision, dict(mode=precision, weights=precision,
                    activations=precision, accumulation="PyTorch CUDA operator-dependent; no autocast", exceptions=[])), cases=[])
    inputs, expected = {}, {}
    with torch.inference_mode():
        for case in spec["cases"]:
            name = case["name"]
            progress.mark("case_begin", name)
            if benchmark:
                values = {k: torch.from_numpy(prepared[name + "__" + k].copy()) for k in ("input_ids", "attention_mask")}
                prefix = torch.from_numpy(prepared[name + "__decoder_input_ids"].copy())
                target = case["target_id"]
            else:
                tokenizer.src_lang = case["src"]
                values = tokenizer(case["texts"], padding=True, truncation=True, max_length=spec["max_source_tokens"], return_tensors="pt")
                if "token_length" in case:
                    values = {k: v[:, :case["token_length"]].clone() for k, v in values.items()}
                    values["input_ids"][:, -1] = architecture["eos_token_id"]
                pad = min(case.get("extra_padding", 0), spec["max_source_tokens"] - values["input_ids"].shape[1])
                values = {k: torch.nn.functional.pad(v, (0, pad), value=architecture["pad_token_id"] if k == "input_ids" else 0) for k, v in values.items()}
                target = case["target_id"] = tokenizer.convert_tokens_to_ids(case["tgt"])
                prefix = torch.tensor([[architecture["decoder_start_token_id"], target]] * len(case["texts"]))
            inputs.update({name + "__" + k: v.numpy() for k, v in {**values, "decoder_input_ids": prefix}.items()})
            if case["numerical"]:
                progress.mark("forward", name)
                output = model(**{k: v.cuda() for k, v in values.items()}, decoder_input_ids=prefix.cuda(), use_cache=False)
                progress.mark("forward_readback", name)
                expected[name + "__encoder"] = output.encoder_last_hidden_state.float().cpu().numpy()
                expected[name + "__logits"] = output.logits[:, -1].float().cpu().numpy()
            samples, first = [], None
            torch.cuda.reset_peak_memory_stats()
            for repeat in range((spec["timed_repeats"] if case["numerical"] else 1) + 1):
                progress.mark("generate", name, repeat)
                start = time.perf_counter()
                tokens = model.generate(**{k: v.cuda() for k, v in values.items()}, forced_bos_token_id=target,
                    max_new_tokens=case["max_new_tokens"], do_sample=False, num_beams=1, use_cache=True).cpu()
                elapsed = time.perf_counter() - start
                progress.mark("generate_synchronize", name, repeat)
                start = time.perf_counter()
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - start
                if first is None:
                    first, cold = tokens, elapsed
                else:
                    if not torch.equal(first, tokens):
                        raise ValueError("CUDA nondeterminism")
                    samples.append(elapsed)
            expected[name + "__tokens"] = first.numpy()
            meta["cases"].append(dict(name=name, seconds=samples, first_seconds=cold, translations=tokenizer.batch_decode(first,
                skip_special_tokens=True), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            progress.mark("case_complete", name)
    progress.mark("write_outputs")
    meta["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    meta["suite_sha256"] = receipt["suite_sha256"] = suite_hash(spec)
    if benchmark:
        np.savez_compressed(root / "actual.npz", **expected)
        write_json(root / "measurements.json", meta)
        result = assess(args)
        from sacrebleu.metrics import CHRF
        original = json.loads((private / "reference.json").read_text())
        hypotheses = [text for c in meta["cases"] for text in c["translations"]]
        translations = [text for c in original["cases"] for text in c["translations"]]
        result["reference_agreement"] = dict(chrf_vs_fp32=CHRF(word_order=2).corpus_score(hypotheses, [translations]).score,
                                            scope="short-output agreement with FP32; not human-reference quality")
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
    if spec.get("version", 1) >= 2:
        validate_receipt(json.loads((public / "checkpoint_receipt.json").read_text()), identity, selected, spec)
    verified = verify_weights(args.weights, identity["model_key"])
    progress.mark("import_runtime", initializing=True)
    import numpy as np
    import torch
    import ttnn
    from torch.utils._python_dispatch import TorchDispatchMode
    require_tt_precision(args.precision, ttnn)
    class RejectCPUCompute(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func).startswith(("aten.mm.", "aten.bmm.", "aten.addmm.", "aten.matmul.", "aten.linear.",
                "aten.embedding.", "aten.native_layer_norm.", "aten.layer_norm.", "aten._softmax.",
                "aten.softmax.", "aten.gelu.", "aten.scaled_dot_product")):
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
            ids, mask, prefix = (inputs[name + "__" + k] for k in ("input_ids", "attention_mask", "decoder_input_ids"))
            if case["numerical"]:
                progress.mark("forward", name)
                originals = dict(input_ids=ids, attention_mask=mask, decoder_input_ids=prefix)
                supplied = {key: value.copy() for key, value in originals.items()}
                with RejectCPUCompute():
                    result = backend.forward(supplied["input_ids"], supplied["attention_mask"],
                                             supplied["decoder_input_ids"])
                progress.mark("forward_synchronize", name)
                ttnn.synchronize_device(device)
                validate_input_preservation(originals, supplied)
                logits = np.asarray(result["logits"])
                if logits.shape != (*prefix.shape, config["vocab_size"]):
                    raise ValueError("forward logits must be [B,T,V]")
                arrays[name + "__encoder"], arrays[name + "__logits"] = np.asarray(result["encoder"]).copy(), logits[:, -1].copy()
            first, samples = None, []
            for repeat in range((spec["timed_repeats"] if case["numerical"] else 1) + 1):
                progress.mark("generate_pre_synchronize", name, repeat)
                ttnn.synchronize_device(device)
                progress.mark("generate", name, repeat)
                before, start = calls[0], time.perf_counter()
                supplied = dict(input_ids=ids.copy(), attention_mask=mask.copy())
                with RejectCPUCompute():
                    tokens = np.asarray(backend.generate(supplied["input_ids"], supplied["attention_mask"],
                                                        case["target_id"], case["max_new_tokens"]))
                elapsed = time.perf_counter() - start
                progress.mark("generate_synchronize", name, repeat)
                start = time.perf_counter()
                ttnn.synchronize_device(device)
                elapsed += time.perf_counter() - start
                validate_input_preservation(dict(input_ids=ids, attention_mask=mask), supplied)
                if calls[0] == before or tokens.dtype.kind not in "iu":
                    raise ValueError("missing TT compute or invalid tokens")
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
    config = selected["architecture"]
    strict = spec.get("version", 1) >= 2
    inputs = np.load(public / "inputs.npz", allow_pickle=False) if strict else None
    actual, expected = np.load(out / "actual.npz", allow_pickle=False), np.load(private / "oracle.npz", allow_pickle=False)
    measured = json.loads((out / "measurements.json").read_text())
    entries = measured["cases"]
    measurements = {c["name"]: c for c in entries}
    failures, checks, canonical = [], [], {}
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
    keys = {c["name"] + "__" + k for c in spec["cases"] for k in (("tokens", "encoder", "logits") if c["numerical"] else ("tokens",))}
    if set(actual.files) != keys or set(expected.files) != keys:
        failures.append("missing/unexpected arrays")
    if len(entries) != len(spec["cases"]) or set(measurements) != {c["name"] for c in spec["cases"]}:
        failures.append("missing/duplicate timings")
    for case in spec["cases"]:
        name, check = case["name"], {"name": case["name"]}
        try:
            tokens = actual[name + "__tokens"]
            if tokens.ndim != 2 or tokens.shape[0] != len(case["texts"]) or tokens.dtype.kind not in "iu":
                raise ValueError("invalid tokens shape/dtype")
            canonical[name] = [canonical_tokens(r, case["target_id"], case["max_new_tokens"], config) for r in tokens]
            oracle = [canonical_tokens(r, case["target_id"], case["max_new_tokens"], config) for r in expected[name + "__tokens"]]
            if len(oracle) != len(tokens):
                raise ValueError("oracle token batch shape mismatch")
            check["exact_reference_rows"] = sum(a == b for a, b in zip(canonical[name], oracle))
            check["rows"] = len(tokens)
            if case["numerical"]:
                for key in ("encoder", "logits"):
                    if actual[name + "__" + key].dtype.kind != "f":
                        raise ValueError(key + " must contain floating-point values")
                    a, b = (data[name + "__" + key].astype(np.float64) for data in (actual, expected))
                    if strict:
                        shape = ((len(tokens), inputs[name + "__input_ids"].shape[1], config["d_model"])
                                 if key == "encoder" else (len(tokens), config["vocab_size"]))
                        if a.shape != shape or b.shape != shape:
                            raise ValueError(key + " dimensions disagree with pinned architecture/input")
                    if a.shape != b.shape or a.shape[0] != len(tokens) or not np.isfinite(a).all() or not np.isfinite(b).all():
                        raise ValueError(key + " shape/finite check")
                    axes = tuple(range(1, a.ndim))
                    error = float(np.max(np.sqrt(np.mean((a - b) ** 2, axis=axes)) / np.maximum(1e-12, np.sqrt(np.mean(b ** 2, axis=axes)))))
                    if not math.isfinite(error):
                        raise ValueError(key + " nonfinite NRMSE")
                    check[key + "_max_row_nrmse"] = error
                    if error > GATES["nrmse"]:
                        raise ValueError(key + " NRMSE exceeds 0.04")
            if "compare_to" in case and canonical[name] != [canonical[case["compare_to"]][i] for i in case["compare_rows"]]:
                raise ValueError("batch/order/padding/revisit semantics changed")
            samples = measurements[name]["seconds"]
            if not samples or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError("invalid timing samples")
            if strict and len(samples) != (spec["timed_repeats"] if case["numerical"] else 1):
                raise ValueError("timing repeat count disagrees with prepared suite")
            cold = measurements[name].get("first_seconds")
            if strict and (type(cold) not in (int, float) or not math.isfinite(cold) or cold <= 0):
                raise ValueError("invalid first-call timing")
            generated = sum(len(row) - 1 for row in canonical[name])
            check.update(p50_seconds=statistics.median(samples), timing_samples=len(samples), seconds=samples,
                         first_seconds=cold, generated_tokens=generated,
                         tokens_per_second=generated / statistics.median(samples))
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            failures.append(name + ": " + str(exc))
        checks.append(check)
    quality = {}
    if spec["stage"] == "full" and not failures:
        from transformers import AutoTokenizer
        from sacrebleu.metrics import CHRF
        tokenizer = AutoTokenizer.from_pretrained(selected["repo_id"], revision=selected["revision"], trust_remote_code=False, local_files_only=True)
        labels = json.loads((private / "labels.json").read_text())
        reference = {c["name"]: c for c in json.loads((private / "reference.json").read_text())["cases"]}
        for src, tgt in [("eng_Latn", l) for l in LANGUAGES] + [(l, "eng_Latn") for l in LANGUAGES]:
            subset = [c["name"] for c in spec["cases"] if c.get("quality") and (c["src"], c["tgt"]) == (src, tgt)]
            hyp, ref, human = [], [], []
            for name in subset:
                hyp.extend(tokenizer.batch_decode(canonical[name], skip_special_tokens=True))
                ref.extend(reference[name]["translations"])
                human.extend(labels[name])
            metric = CHRF(word_order=2)
            if len(human) != 1012 or len(hyp) != len(human) or len(ref) != len(human):
                failures.append(src + "->" + tgt + ": incomplete full quality coverage")
                continue
            score, baseline = (metric.corpus_score(x, [human]).score for x in (hyp, ref))
            quality[src + "->" + tgt] = dict(chrf=score, cuda_chrf=baseline, rows=len(human))
            if not math.isfinite(score) or not math.isfinite(baseline) or baseline - score > GATES["max_chrf_loss"]:
                failures.append(src + "->" + tgt + ": chrF++ loss exceeds 0.5")
    nrmse = [v for check in checks for k, v in check.items() if k.endswith("_nrmse")]
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
    ref.add_argument("--model", default="600m", choices=("600m", "1.3b-distilled", "3.3b"))
    for flag in ("corpus", "out"):
        ref.add_argument("--" + flag, required=True)
    for command, flags in (("candidate", ("input", "output", "weights")), ("assess", ("input", "oracle", "output")),
                           ("reference-benchmark", ("input", "oracle", "output"))):
        child = sub.add_parser(command)
        for flag in flags:
            child.add_argument("--" + flag, required=True)
        if command != "assess":
            child.add_argument("--precision", choices=PRECISIONS if command == "candidate" else ("fp32", "fp16", "bf16"), default="bf16")
        if command == "reference-benchmark":
            child.add_argument("--model", choices=("600m", "1.3b-distilled", "3.3b"))
    args = parser.parse_args()
    result = {"reference": reference, "reference-benchmark": reference, "candidate": candidate, "assess": assess}[args.command](args)
    if args.command in ("assess", "reference-benchmark"):
        print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
        sys.exit(0 if result["passed"] else 1)
