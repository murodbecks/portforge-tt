# Parakeet TDT → Blackhole port

Build one correct, fast, configurable TTNN implementation for
`nvidia/parakeet-tdt-0.6b-v3` (0.6B, FastConformer encoder + TDT decoder,
CC-BY-4.0 weights), packaged in tt-metal style for an upstream `tt-metal` PR.
Read CONTRACT.md for the exact interface and acceptance envelope; read
LESSONS.md before the first device job. This is a shared machine — never touch
other users' files, never reset cards, keep everything inside this campaign's
workspace.

1. Read TT_GUIDE.md, SOURCES.md, BENCHMARK.md and LESSONS.md first; inspect the installed
   NeMo/transformers sources and the pinned checkpoint. Save findings and
   citations in RESEARCH.md. Explain the FastConformer subsampling conv stack,
   depthwise convolutions, relative/positional handling, the TDT joint decode
   loop (durations 0-4, max 10 symbols/step) and precision choices before
   implementing.
2. BASELINE FIRST: record measured baselines on the fixed suites from the TT
   host — PyTorch FP32 (transformers reference) and, if the installed stack
   allows, an ONNX/forge path. The FP32/BF16 CUDA reference runs on the A100
   provide the NVIDIA comparison timings (RTF). All optimization claims cite
   matched baseline artifacts; beating the A100 is a stretch goal, not a gate.
3. Write backend.py and component checks. Use `port_run` for isolated
   experiments. The pinned checkpoint is mounted under `/weights`; mel
   preprocessing is deterministic CPU policy (declared, like tokenization in
   the NLLB port). Do not install packages or reset devices.
4. `port_evaluate(stage="smoke")` → FP32 oracle check → `bringup`.
   Establish repeated-call memory behavior with bounded diagnostics.
5. Pass `bringup` before optimizing. Profile where time and memory go
   (subsampling convs, attention, the autoregressive TDT loop, host round
   trips per emitted token — this decode loop is the usual TT risk), then
   compare changes on identical workloads. BF16 first; FP32 for diagnosis;
   BFP8_B as a separate block-float experiment. Declare precision exceptions.
6. REPORT.md with correctness evidence, reproducible commands, baseline vs
   optimized latency/memory, limitations and sources. Keep STATE.md short and
   current. Prepare branch/commit/PR-body materials early (lesson 9).
7. Produce the tt-metal-style deliverable from CONTRACT.md; portable tests;
   the human operator reviews, signs and submits.
8. Call `port_submit` only after current-source bringup passes.

Only `port_write` persists edits. `port_run/evaluate/status/log/submit`
are the workflow tools. Page truncated logs with `port_log`; do not rerun
jobs to recover output. Checkpoint within ~8 minutes (strong target — overrun
only for clearly promising in-flight work, with a STATE.md progress note; hard
cap 30 minutes/turn). Compute must run on TT with real weights; preserve
padding/length masks, batch independence and greedy decode determinism. Do not
exploit the corpus, cache transcripts, or change the contract. Variable-length
audio beyond the frozen envelope and streaming/chunked inference are explicit
out-of-scope extensions, reported honestly rather than implied.
