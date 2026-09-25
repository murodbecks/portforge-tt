# Benchmark: what the port is measured on, and why

A **porting benchmark**, not an ASR-leaderboard claim. It answers: (1) is the
TT implementation numerically faithful to the FP32 oracle (encoder states +
greedy tokens), (2) does it preserve batch/length/revisit semantics, (3) is it
faster relative to its own recorded baseline — with the A100 reference timings
as the NVIDIA comparison context. Absolute transcription quality is Parakeet's
problem; we gate only *agreement with the FP32 reference*.

## Frozen corpus (`corpus.json`, canonical sha256
`62fb9c517719931999ccd72b9c097e889e843aaf8345567f62f16e95e8248d02`)

| Clips | Origin | Why |
|---|---|---|
| 15 × LibriSpeech dev-clean (2.2–16.4 s, one per speaker) | [openslr 12](https://www.openslr.org/resources/12/dev-clean.tar.gz), CC-BY-4.0 | Real clean read speech, the classic ASR fixture, human transcripts for WER; one clip per speaker avoids speaker overfitting; duration spread exercises subsampling/lengths. |
| syn-silence (2 s), syn-noise (3 s), syn-tone (1 kHz, 3 s), syn-short (0.4 s) | deterministic stdlib-generated WAV | Degenerate inputs: zero energy, broadband noise, pure tone, sub-second clip — catch NaN paths, length edge cases and blank-only decode behavior. |

Audio is embedded base64 inside the frozen JSON (3.1 MB, 124 s total) so every
campaign and reference run sees byte-identical data. The evaluator prepares
128-bin log-mel features once (public `inputs.npz`); candidates never see raw
audio or transcripts.

## Stages and gates

| Stage | Cases | Contents |
|---|---|---|
| `smoke` | 1 | shortest clean clip (~2 s) |
| `bringup` | 8 | batch(2)/single/reorder consistency, ~16 s long clip, ~2 s short clip, synthetic silence |
| `full` | 22 | all 15 libri clips (WER-gated, held-out human transcripts), batch4 + reverse, revisit, silence, long |

- Encoder max-row NRMSE ≤0.04 vs FP32 oracle (every numerical case).
- Greedy tokens exactly match the FP32 oracle rows (padding stripped);
  repeated identical calls bit-identical; batch/reorder/revisit consistent.
- Full only: WER inflation vs the FP32 reference transcripts ≤2%.
- Diagnostics (non-gating): per-case RTF (audio seconds / wall seconds),
  first-call latency, A100 comparison context, memory snapshots.

## Why not the big ASR benchmarks

`open_asr_leaderboard` / Common Voice / multilingual evals measure model
accuracy across domains and languages; using them as the inner porting gate
would repeat the NLLB mistake of waiting on evaluation instead of engineering.
LibriSpeech dev-clean is the bounded, license-clean (CC-BY-4.0) sample with
human labels; multilingual/long-audio coverage is an explicit out-of-scope
extension, as is streaming. Extend the corpus deliberately (new hash, new
references, documented provenance) — never by editing frozen clips.
