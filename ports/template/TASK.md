# <Model> → Tenstorrent port

<!-- The worker's ordered plan. Keep the order; replace TODOs with model detail. -->

Build one correct, fast, configurable TTNN implementation of `TODO/hf-repo-id`,
packaged in tt-metal style for an upstream PR. CONTRACT.md defines the
interface and acceptance envelope. Read LESSONS.md before the first device job.

1. Read TT_GUIDE.md, SOURCES.md, BENCHMARK.md and LESSONS.md. Inspect the
   pinned checkpoint and the reference implementation. Save concise findings
   and citations in RESEARCH.md. TODO: list the architecture details that must
   be understood before implementing (e.g. masking, positions, normalization,
   decoding rules).
2. Baseline first: measure PyTorch on the TT host's CPU (and ONNX/forge if
   installed) on the smoke and bringup suites. Record commands, samples and job
   IDs in BASELINE_*.md.
3. Write `backend.py` and small component checks. Use `port_run` for bounded
   experiments. Edit files only with `port_write`; files written inside a job
   are discarded.
4. Run `port_evaluate(stage="smoke")`, collect results with
   `port_status(compact=True)`, and fix one hypothesis at a time.
5. Pass `bringup` before optimizing. Profile, then compare changes on identical
   workloads against the recorded baselines. Start with BF16, compare FP32 when
   diagnosing numerics, and treat BFP8_B as a separate experiment.
6. Write REPORT.md: correctness evidence, commands, baseline versus optimized
   latency and memory, limitations and sources. Keep STATE.md short: completed
   work, current hypothesis, pending job IDs, next action.
7. Build the tt-metal package from CONTRACT.md and prepare the branch name,
   commit message and PR body early.
8. Rerun smoke and bringup on the final source, finish all documents, then call
   `port_submit` exactly once. Every file becomes read-only after submitting.
