# Benchmark: what the port is measured on, and why

The harness benchmark is a **porting benchmark**, not a forecasting-leaderboard
claim. It answers three questions quickly, on real data, with held-out ground
truth: (1) is the TT implementation numerically faithful to the FP32 oracle,
(2) does it preserve batch/mask/revisit semantics, and (3) is it getting
faster relative to its own recorded baseline. Task-level forecast quality is
measured only at the full gate, and only as *agreement with the FP32 reference*
— the model's absolute accuracy is Chronos-2's problem, not the port's.

## What is in the frozen corpus (`corpus.json`)

| Series | Origin | Why it is here |
|---|---|---|
| `etth1_hufl…etth1_ot` (7) | ETTh1, last 1536 hourly rows (Jun–Aug 2018), [ETDataset](https://github.com/zhouhaoyi/ETDataset), MIT | Real-world electricity-transformer telemetry; the classic long-horizon forecasting fixture used by PatchTST/iTransformer papers and by Tenstorrent's own time-series bounty issues (#32139/#32140). Different scales and regimes per column stress normalization and arcsinh scaling honestly. |
| `syn_sine_mixed` | deterministic: 168/24-hour sines + seeded noise (`Random(1729)`) | Smooth doubly-periodic signal; easy to eyeball, checks that periodic structure survives patching. |
| `syn_arcsinh_stress` | deterministic: `sinh(4·sin)` + tiny noise (`Random(4104)`) | Alternating tiny/huge magnitudes (|x| up to ~27). Directly stresses `use_arcsinh` scaling, FP range behavior and BFP8_B shared-exponent loss. |
| `syn_constant` | `7.0` × 1536 | Degenerate input: zero variance after normalization; catches NaN/div-by-zero paths in scaling and quantile heads. |

All values are frozen in the file (canonical-JSON sha256
`f0b08df4fe80385da1d3e9a05e3779708c4e808b2da581aa974fea9505d64ae3`); synthetic
series are frozen outputs, not regenerated, so every campaign and every
reference run sees byte-identical data.

## Workload shape and stage sizes

Fixed envelope (per CONTRACT.md): batch 1–4, context ≤512, horizon ≤64,
univariate, the model's 21 native quantiles. Windows carve disjoint
`[context | future]` slices; futures are ground truth for the full-stage task
metric and stay in the private oracle directory.

| Stage | Cases | Contents | Wall-time intent |
|---|---|---|---|
| `smoke` | 1 | ETTh1 OT, context 512 → horizon 64 | seconds; the innermost bring-up loop |
| `bringup` | 8 | batch(2)/single/reorder consistency, masked tail (16 missing), boundary contexts 64/65, constant series | a minute or two; the correctness gate before any optimization |
| `full` | 35 | 9 series × 3 windows (ctx 512/128/512, horizon 64/16/64) + batch4 + masked-holes + constant + behavior/revisit cases; 30 carry held-out futures | minutes on GPU or TT; run once, on frozen source, via `submit` |

Timing: 1 warmup + 1 measured repeat per case in short stages, 3 measured
repeats in full; protocol `cpu_arrays_to_cpu_arrays_with_sync_excluding_load_
preprocessing_progress_v1`.

## Gates

- `quantile_max_row_nrmse ≤ 0.04` vs the FP32 oracle (per batch row, over the
  full [horizon × 21-quantile] block) — every numerical case.
- Behavior: same series forecast through different batch shapes/orders must
  agree within NRMSE ≤ 0.01; repeated identical calls must be bit-identical.
- Full stage only: weighted quantile-loss (mean pinball over steps and all 21
  levels, scaled by mean |y|) on held-out futures may inflate by ≤2% versus
  the FP32 reference's own forecasts. This is agreement with the reference,
  not a leaderboard score.
- Diagnostics (non-gating): quantile-crossing counts, first-call latency,
  forecasts/second, memory snapshots.

## Why not the big public benchmarks

- **GIFT-Eval / fev-bench / Chronos Benchmark II**: these are where Chronos-2's
  zero-shot SOTA claims live, but they span tens of datasets (many tens of GB),
  need their own loaders/scorers, and measure model accuracy rather than port
  fidelity. Using them as the porting loop's inner gate would repeat the NLLB
  mistake of spending turns waiting on evaluation instead of engineering. They
  are the right *post-port* validation if we later want an accuracy claim, and
  SOURCES.md links them for context.
- **Full ETTh1 (17k rows) / ETTm / Electricity / Traffic**: bigger windows add
  wall time without changing what they can detect about the TT port at this
  envelope; the frozen 1536-point tails keep every suite short while remaining
  real data with published provenance.
- **Random-noise-only corpora**: cheap, but hide exactly the failure modes we
  care about (periodicity, magnitude spread, degenerate variance) — hence the
  three purpose-built synthetic series instead.

If a later phase wants broader coverage, extend the corpus file and pins
deliberately (new sha256, new references, documented provenance) — never by
quietly editing series the agent has already been evaluated against.
