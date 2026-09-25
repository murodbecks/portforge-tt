# Chronos-2 backend contract

Write `backend.py` from scratch for pinned `amazon/chronos-2` (model key
`chronos2`, Apache-2.0). Use configuration values rather than hardcoding
architecture dimensions; the registry in `models.json` is authoritative.

```python
def create_backend(weights_path, config, device, *, precision="bf16"):
    # Return an object with the methods below. The harness owns the TTNN device.
    ...

class Backend:
    # Required instance declaration; update it for the requested precision.
    precision_policy = {"mode": "bf16", "weights": "bf16", "activations": "bf16",
                        "accumulation": "fp32", "exceptions": []}

    def forecast(self, past_values, past_observed_mask, prediction_length):
        # CPU numpy float32 inputs [B,T] and [B,T]; 1.0 = observed, 0.0 = missing.
        # Masked cells hold 0.0 placeholders; the reference maps them to the
        # package's internal missing-value semantics (NaN) itself. Never mutate inputs.
        # prediction_length is 1..64. Return CPU numpy float32:
        # {"quantiles": [B, prediction_length, Q]} in the config's quantile order
        # (21 native levels). Values are point forecasts, not samples; deterministic.
        ...
```

Load the pinned official safetensors checkpoint (CPU conversion and patch/value
preparation are allowed); learned inference must run on TT. The public oracle
comparison happens through the `chronos_forecasting` package's
`Chronos2Pipeline.predict` (univariate, `inputs=[B,1,T]`, direct multi-patch
output; no sampling). Batched rows must remain independent in the default mode:
cross-learning/group attention across unrelated series rows is NOT part of this
contract; a later multivariate/covariate extension is separate work with its own
validation. Do not close/open another device.
Optional `DEVICE_OPTIONS`: `trace_region_size` 0–128 MiB (potentially per bank),
`l1_small_size` 0–128 KiB, `num_hw_cqs` 1–2. Return CPU-visible results.

Evaluation selects `bf16` (default), `fp32`, `fp16`, or `bfp8_b` explicitly and
passes it to the factory. Declare actual dominant weight/activation formats,
accumulation and every higher-precision exception in `precision_policy`; primary
weight format and mode must match the request. Mixed activations require an
explanation in `exceptions`. These are auditable declarations, not automatic
proof of all tensor formats. Unsupported modes must fail clearly, never silently
substitute another format. Installed TTNN may not expose native FP16. TT
`bfloat8_b` is block floating point with shared exponents, **not IEEE FP8**.
The reference pipeline exposes FP32 and BF16 only; the FP16 diagnostic study is
therefore unavailable for this model.

The immutable correctness oracle stays FP32 on the configured reference host
(CUDA preferred; CPU allowed and recorded in the receipt). CUDA diagnostic
comparisons replay identical smoke/bringup inputs in FP32/BF16; they do not
replace that oracle or relax gates. Sensitive arithmetic (arcsinh scaling,
patch normalization, quantile-head accumulation) is a legitimate place for
declared FP32 exceptions with measured justification.

Required envelope: batch 1–4, context up to 512 points, prediction length up to
64. Preserve masks, row order and repeated-call semantics. Quantile forecast
error must be finite and max-row NRMSE ≤0.04 against the independent FP32
reference on every numerical case. Batch/single/reorder/masked/revisit outputs
for the same series must agree within max-row NRMSE ≤0.01 (float outputs are not
required to match bitwise across different batch shapes, but repeated identical
calls must be exactly deterministic). Quantile crossing may occur numerically and
is reported, not gated.

BASELINE BEFORE OPTIMIZATION: before any optimization claim, record measured
baselines on the fixed suites from the same TT host — (a) PyTorch
(chronos-forecasting, CPU) and, if it works in the installed stack, (b) an
ONNX/forge path. Every speedup claim cites the matched baseline artifact
(`baselines/` receipts), the same suite, precision and card. TT-vs-TT ratios
never mix with CUDA context numbers.

Benchmark rationale and corpus provenance: BENCHMARK.md. Develop progressively: `smoke` is one real-model numerical forecast (context
512, horizon 64); `bringup` adds eight short batch/padding/mask/length-boundary/
constant cases. Each has one warmup and one measured forecast. Run these before
`full`: the frozen corpus windows (7 real ETTh1 series + 3 deterministic
synthetic series; contexts 64/128/512, horizons 16/64, disjoint windows),
behavior cases and revisits. Full additionally gates task-level weighted
quantile-loss inflation ≤2% versus the FP32 reference forecasts on held-out
futures. Timed numerical cases have three measured repetitions in full; other
cases have one. Early passes are debugging milestones, never acceptance.

For a successful smoke/bringup result, `verdict.passed` is `true` and
`verdict.accepted` is `false`; only a successful full evaluation sets both true.
Collect genuine results with `port_status(compact=True)`, which includes the
structured verdict and request/source/cluster identity. `port_log` retrieves
stdout/stderr only. The isolated `port_run` environment receives public
inputs, not trusted verdicts or oracle outputs. Do not invent an authenticated
receipt mount, copy oracle data into a backend, or treat candidate-authored pass
flags as authority. Freeze the implementation before its real gates; preserve
their job IDs outside source code so recording a result does not change the
source hash. Performance claims must be checked against the controller's
original requests and results.

The run sandbox mounts the controller's public `/models.json` registry and
`/models.py` read-only, plus prepared `/input` files and routed `/weights`
(config.json + model.safetensors). Inspect these with a bounded `port_run`
before assuming a metadata schema. Registry entries contain official file
hashes/sizes and architecture constraints.

Timing includes array transfer, host control, learned compute and
synchronization; excludes weight loading and progress-file writes. Report
matched workload speedups, host RSS and allocator memory snapshots with their
scopes. Do not call three samples a tail-latency SLA. Do not exploit known test
inputs, precompute forecasts, run hidden CPU inference or alter the trusted
evaluator. The full corpus is a small frozen sample for bounded research
acceptance — not GIFT-Eval/fev-bench leadership, production certification,
multivariate/covariate support, streaming or commercial SLAs.

## tt-metal deliverable

The final artifact must be a production-style tt-metal model package suitable
for an upstream PR to `tenstorrent/tt-metal` under `models/experimental/chronos2`
(follow the layout of existing model demos such as ModernBERT, SpeechT5 and the
time-series bring-ups): `tt/` (TTNN implementation), `demo/` (standalone
forecast CLI: CSV in → quantile CSV out, with explicit checkpoint/device/
precision arguments), `reference/` (FP32 torch reference utilities),
`tests/` (pytest, importable without CHIA mounts, configurable fixtures),
`benchmarks/` (reproducible measurement scripts), `docs/`, top-level README with
model card links, measured evidence and limitations; SPDX headers on source
files; no vendored third-party code without license attribution (Apache-2.0
compatible only). The worker prepares branch, commit message and PR body; the
human operator reviews, signs (DCO) and submits — the agent never pushes or
opens the PR itself.
