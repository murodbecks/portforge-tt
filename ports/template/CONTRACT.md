# <Model> backend contract

<!-- Replace every TODO. Agents read this file first; it is the acceptance envelope.
     Keep the generic sections unless your evaluator changes them. -->

Write `backend.py` from scratch for the pinned `TODO/hf-repo-id` (model key
`my-model`, license TODO). Read configuration values rather than hardcoding
architecture dimensions; the registry in `models.json` is authoritative.

```python
def create_backend(weights_path, config, device, *, precision="bf16"):
    # Return an object with the methods below. The harness owns the TTNN device.
    ...

class Backend:
    # Required instance declaration; update it for the requested precision.
    precision_policy = {"mode": "bf16", "weights": "bf16", "activations": "bf16",
                        "accumulation": "fp32", "exceptions": []}

    def TODO_method(self, TODO_inputs):
        # TODO: exact input arrays (dtype, shape, masks) and returned arrays,
        # matching forward_case() in evaluate.py. Never mutate inputs.
        ...
```

Load the pinned official checkpoint (CPU weight conversion and deterministic
preprocessing are allowed); learned inference must run on TT. The oracle is
TODO: the official implementation and entry point it runs. Batched rows must
stay independent. Do not open or close another device. Optional
`DEVICE_OPTIONS`: `trace_region_size` 0–128 MiB, `l1_small_size` 0–128 KiB,
`num_hw_cqs` 1–2. Return CPU-visible results.

## Precision

Evaluation selects `bf16` (default), `fp32`, `fp16`, or `bfp8_b` explicitly and
passes it to the factory. Declare the actual dominant weight and activation
formats, accumulation, and every higher-precision exception in
`precision_policy`; the weight format and mode must match the request. Mixed
activations require an explanation in `exceptions`. Unsupported modes must fail
clearly and never silently substitute another format. TT `bfloat8_b` is block
floating point with shared exponents, not IEEE FP8.

The correctness oracle stays FP32 on the reference host (CUDA preferred, CPU
recorded). CUDA BF16 comparisons are context only; they never replace the
oracle or relax gates.

## Envelope and gates

TODO: required envelope (batch sizes, lengths, horizons, languages, ...), and
what is explicitly out of scope.

TODO: the numerical gates from `GATES` in evaluate.py, stated in words, for
example: max-row NRMSE ≤0.04 against the FP32 oracle on every numerical case;
consistency cases (single rows, reversed batch, revisit) agree within ≤0.01;
repeated identical calls are bit-identical; task metric on the full stage.

## Baseline before optimization

Before any optimization claim, record measured baselines on the fixed suites
from the same TT host: PyTorch on the host CPU and, if the installed stack
supports it, an ONNX/forge path. Every speedup cites the matched baseline
artifact, the same suite, precision and card. TT-vs-TT ratios never mix with
CUDA context numbers.

## Stages

`smoke` (one real case) and `bringup` (TODO: about eight cases covering batch,
padding, masks, length boundaries and edge inputs) have one warmup and one
measured call each. `full` is the frozen held-out corpus plus consistency
cases, run once on frozen source through `port_submit`. Early passes are
debugging milestones, never acceptance: `verdict.accepted` is true only after
a successful full evaluation. Corpus provenance: BENCHMARK.md.

## Evidence and sandbox

Collect results with `port_status(compact=True)`; `port_log` returns
stdout/stderr only. `port_run` jobs see public inputs, never verdicts or oracle
outputs, and mount `/models.json`, `/models.py`, prepared `/input` files and
routed `/weights` read-only. Do not exploit known test inputs, precompute
outputs, run hidden CPU inference or alter the trusted evaluator. Timing
includes transfers, host control, learned compute and synchronization, and
excludes weight loading.

## tt-metal deliverable

A tt-metal model package for an upstream PR under
`models/experimental/<model>`: `tt/` (TTNN implementation), `demo/` (CLI),
`reference/` (FP32 reference utilities), `tests/` (pytest, importable without
CHIA mounts), `benchmarks/`, `docs/`, and a README with model-card links,
measured evidence and limitations. SPDX headers on source files; only
Apache-2.0-compatible code. The worker prepares branch, commit message and PR
body; the operator reviews, signs and submits.
