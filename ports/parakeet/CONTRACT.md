# Parakeet TDT backend contract

Write `backend.py` from scratch for pinned `nvidia/parakeet-tdt-0.6b-v3` (model
key `parakeet`, CC-BY-4.0 weights). Use configuration values rather than
hardcoding dimensions; the registry in `models.json` is authoritative.

```python
def create_backend(weights_path, config, device, *, precision="bf16"):
    # Return an object with the methods below. The harness owns the TTNN device.
    ...

class Backend:
    precision_policy = {"mode": "bf16", "weights": "bf16", "activations": "bf16",
                        "accumulation": "fp32", "exceptions": []}

    def encode(self, mel, mel_lengths):
        # CPU numpy float32 [B,T,128] log-mel frames (time-major) and int64 [B]
        # true frame counts (frames are padded to T; padding is zero-masked). Return CPU numpy float32:
        # {"encoder": [B, ceil(T_i/8) padded to common length, hidden=1024]}
        # respecting per-row lengths; padded rows' trailing frames are ignored
        # by the evaluator. Never mutate inputs.
        ...

    def transcribe(self, mel, mel_lengths):
        # Greedy TDT decode (durations 0..4, at most 10 symbols per step,
        # blank=8192, pad=2, vocab 8193). Return CPU numpy integers [B,L];
        # rows may have different unpadded lengths. Deterministic: repeated
        # identical calls must produce bit-identical tokens.
        ...
```

Deterministic STFT/log-mel preprocessing is host CPU policy (declared), like
tokenization in the translation port; learned inference must run on TT. The
oracle uses the same public `transformers` implementation (`ParakeetForTDT`)
via `AutoProcessor`/`generate`. The `.nemo` and `.gguf` repository variants are
not part of the pinned manifest. Load the pinned safetensors with
`torch.load(..., map_location="cpu", weights_only=True)` or the safetensors
reader; CPU weight conversion is allowed.
Optional `DEVICE_OPTIONS`: `trace_region_size` 0–128 MiB, `l1_small_size`
0–128 KiB, `num_hw_cqs` 1–2.

Evaluation selects `bf16` (default), `fp32`, `fp16`, or `bfp8_b` explicitly
and passes it to the factory. Declare actual dominant weight/activation
formats, accumulation and every higher-precision exception in
`precision_policy`. Unsupported modes must fail clearly, never silently
substitute another format. TT `bfloat8_b` is block floating point, **not IEEE
FP8**. The transformers reference exposes FP32/BF16 only.

The immutable correctness oracle stays FP32 on the reference host (A100
preferred; CPU recorded). Gates: encoder max-row NRMSE ≤0.04 vs the FP32
oracle on every numerical case; greedy tokens must exactly match the FP32
oracle rows (padding stripped); batch/single/reorder/revisit semantics
preserved; repeated calls deterministic. Full additionally gates WER inflation
≤2% vs the FP32 reference transcripts on held-out LibriSpeech clips (word
error rate vs the human transcripts). Timing protocol:
`mel_features_to_tokens_with_sync_excluding_load_preprocessing_progress_v1`;
the A100 reference's per-case seconds are the NVIDIA comparison context —
report matched TT-vs-TT speedups separately from TT-vs-A100 context and never
mix the two ratios.

Required envelope: batch 1–4, clips up to 32 s (≤~4000 mel frames), greedy
decode. Baselines before optimization claims: PyTorch FP32 on the TT host plus
the recorded A100 reference timings; a speedup without a preserved matched
baseline artifact is invalid.

Stages: `smoke` = one short clip; `bringup` = 8 cases (batch 2 + single/reorder
consistency, ~16 s long clip, ~2 s short clip, synthetic silence); `full` = all
15 LibriSpeech clips + batch4 + behavior/revisit + edge cases, three timed
repeats. `verdict.passed` vs `accepted` semantics, receipts, freeze-before-gate
rules and anti-cheat rules are identical to the previous campaigns' contracts.

## tt-metal deliverable

`models/experimental/parakeet` in tt-metal style (`tt/ demo/ reference/ tests/
benchmarks/ docs/`, SPDX headers, README with measured evidence and the
CC-BY-4.0 weight license note). Standalone CLI: 16 kHz wav/flac in →
transcript out with explicit checkpoint/device/precision arguments. The worker
prepares branch/commit/PR body; the human operator reviews, DCO-signs and
submits.
