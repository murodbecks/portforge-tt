# NLLB backend contract

Write `backend.py` from scratch. Start with pinned `facebook/nllb-200-distilled-600M`;
use configuration values rather than hardcoding architecture dimensions. Larger
checkpoints need separate references and validation before claiming support.

```python
def create_backend(weights_path, config, device, *, precision="bf16"):
    # Return an object with the methods below. The harness owns the TTNN device.
    ...

class Backend:
    # Required instance declaration; update it for the requested precision.
    precision_policy = {"mode": "bf16", "weights": "bf16", "activations": "bf16",
                        "accumulation": "fp32", "exceptions": []}

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        # CPU numpy integer inputs [B,S] and [B,T]. Return CPU numpy floats:
        # {"encoder": [B,S,D], "logits": [B,T,V]}.
        ...

    def generate(self, input_ids, attention_mask, target_id, max_new_tokens):
        # CPU numpy integer [B,L]; greedy, beam=1, no sampling.
        # Start=2; first generated token=target_id; EOS=2; padding=1.
        # max_new_tokens includes the forced language token, as in HF generate.
        ...
```

Load the pinned official state dict with `torch.load(..., map_location="cpu",
weights_only=True)`. CPU weight conversion and token selection/control are allowed;
learned inference must run on TT. Do not mutate inputs or close/open another device.
Optional `DEVICE_OPTIONS`: `trace_region_size` 0–128 MiB (potentially per bank),
`l1_small_size` 0–128 KiB, `num_hw_cqs` 1–2. Return CPU-visible results.

Evaluation selects `bf16` (default), `fp32`, `fp16`, or `bfp8_b` explicitly and
passes it to the factory. Declare actual dominant weight/activation formats,
accumulation and every higher-precision exception in `precision_policy`; primary
weight format and mode must match the request. Mixed activations require an
explanation in `exceptions`. These are auditable declarations, not automatic
proof of all tensor formats. Unsupported modes must fail clearly, never silently
substitute another format. In particular, installed TTNN may not expose native
FP16. TT `bfloat8_b` is block floating point with shared exponents, **not IEEE FP8**;
true FP8 requires separately demonstrated runtime/operator support and validation.

The immutable correctness oracle stays FP32. CUDA diagnostic comparisons can load
the same model explicitly in FP32/FP16/BF16 and replay identical smoke/bringup
inputs; these do not replace that oracle or relax gates. Lower-precision results
include numerical/token agreement, latency and allocator memory. Their optional
chrF comparison to FP32 outputs measures agreement, not human translation quality.

Required envelope: batch 1–4, source up to 256 tokens, greedy generation cap up to
64. Preserve masks, padding, row order and repeated-call/cache semantics. Numerical
encoder and prefix-logit error must be finite and max-row NRMSE ≤0.04 against the
independent FP32 CUDA reference. Batch/single/reorder/padding/revisit results must
match exactly; tolerance does not excuse semantic changes.

Develop progressively: `smoke` is one real-model numerical request with four new
tokens; `bringup` adds eight short batch/padding/tile-boundary/Unicode cases. Each
has one warmup and one measured generation. Run these before `full`: all 8,096
FLORES-200 translations across English↔French/Uzbek/Arabic/Chinese, plus behavior,
source boundaries and cache revisits. Full requires per-direction chrF++ loss ≤0.5
versus CUDA. Timed numerical cases have three measured repetitions in full;
other cases have one. Early passes are debugging milestones, never acceptance.

For a successful smoke/bringup result, `verdict.passed` is `true` and
`verdict.accepted` is `false`; only a successful full evaluation sets both true.
Collect genuine results with `port_status(compact=True)`, which includes the
structured verdict and request/source/cluster identity. `port_log` retrieves
stdout/stderr only. The isolated `port_run` environment receives public inputs,
not trusted verdicts or oracle outputs. Do not invent an authenticated receipt
mount, copy oracle data into a backend, or treat candidate-authored pass flags
as authority. Freeze the implementation before its real gates; preserve their
job IDs outside source code so recording a result does not change the source
hash. Performance claims must be checked against the controller's original
requests and results. Short-test success and exact FP32 token agreement are
separate fields; retain any token disagreement.

The run sandbox mounts the controller's public `/models.json` registry and
`/models.py` read-only, plus prepared `/input` files and routed `/weights`.
Inspect these with a bounded `port_run` before assuming a metadata schema.
Legacy 600M version-1 suites identify repo/revision/weight hash without a
checkpoint receipt, model key or manifest hash. Version-2 suites for larger
models include those fields and `/input/checkpoint_receipt.json`. Registry
entries contain official file hashes/sizes and architecture constraints.
The staged `/input/config.json` is serialized from the loaded HF configuration;
its bytes may differ from the official config. File-backed 600M routing mounts
only its weight file under `/weights`; directory-backed larger models also
mount the official config and, for 3.3B, the index and three shards. Verify
consumed bytes and the supported schema; do not invent a missing receipt.

Timing includes fresh encode/decode, transfers, host control and synchronization;
excludes weight loading, tokenization and progress-file writes. Report matched
workload speedups, host RSS and allocator memory snapshots with their scopes.
Do not call three samples a tail-latency SLA. Do not exploit known test inputs,
precompute translations, run hidden CPU inference or alter the trusted evaluator.

The runner enforces external phase deadlines (early 180s, full 300s, initialization
600s) using atomic progress events and an independent whole-job limit. Inspect
the failed phase and device health; do not blindly repeat a hung job. Hardware
failures are not accuracy results. Keep code, research, STATE.md and REPORT.md
reproducible. Full success is bounded research acceptance, not production
certification, all-language coverage, beam/streaming/concurrent-serving support,
or unrestricted commercial licensing. NLLB weights are CC-BY-NC-4.0.
