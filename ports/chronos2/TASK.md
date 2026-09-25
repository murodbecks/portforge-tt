# Chronos-2 → Blackhole port

Build one correct, fast, configurable TTNN implementation for `amazon/chronos-2`
(119.5M parameters, encoder-only, Apache-2.0), packaged in tt-metal style for an
upstream `tt-metal` PR. Read CONTRACT.md for the exact interface and acceptance
envelope; read LESSONS.md before the first device job — those mistakes are
expensive to repeat. Use public research and the pinned installed source.

1. Read TT_GUIDE.md, SOURCES.md, BENCHMARK.md and LESSONS.md; inspect the installed chronos
   package (if present), TTNN transformer implementations and the pinned
   checkpoint. Save concise findings and citations in RESEARCH.md. Explain
   patching, positions/RoPE, masking, reg token, scaling (arcsinh),
   output-patch quantile head and precision choices before implementing.
2. BASELINE FIRST: measure and preserve baselines on the fixed suites from the
   TT host — PyTorch (chronos-forecasting CPU) and, if the installed stack
   allows, an ONNX/forge path. Record BASELINE_*.md with commands, samples and
   job receipts. All later optimization claims compare against these matched
   numbers; a speedup without a recorded baseline is invalid.
3. Write backend.py and useful component checks. Use `port_run` for isolated
   experiments and source inspection. Python, PyTorch and TTNN already exist;
   the installed TT-Metal tree is `/opt/tt-metal`. The pinned checkpoint is
   mounted under `/weights`. Do not install packages or reset devices.
4. Run `port_evaluate(stage="smoke")` for one short real-model request. An
   independent FP32 reference runs on the configured reference host (CUDA
   preferred; CPU recorded). Use `port_status(compact=True)` to collect
   asynchronous results; retain job IDs. Inspect the failed phase, fix one
   hypothesis, and avoid duplicate pending jobs.
5. Pass `bringup` before optimizing. Preserve that source and job as the TT
   baseline anchor. Profile where time and memory go, investigate public
   optimization techniques (traces, batching, layout/sharding), then compare
   changes on identical workloads and settings against the recorded baselines.
   Seek substantial measured gains; there is no required speedup target to game.
   Start with explicit BF16 and compare FP32 when diagnosing numerical
   failures; investigate BFP8_B separately as a block-float experiment.
   Declare precision exceptions and accumulation policy. Never disguise a
   precision change as a same-precision speedup. CUDA FP32/BF16 references
   provide additional context.
6. Write REPORT.md with correctness evidence, reproducible commands, baseline
   versus optimized latency and memory, limitations and sources. Keep STATE.md
   short: completed work, current hypothesis, pending job IDs, next action.
7. Produce the tt-metal-style deliverable from CONTRACT.md: importable,
   configurable backend; standalone forecast CLI; portable tests that import
   actual code and accept configurable fixtures rather than execute Markdown or
   require CHIA's `/input` and `/weights`; package layout with SPDX headers.
   Prepare the branch name, commit message and PR body early (lesson: GitHub
   submission access was discovered missing at the END last time); the human
   operator reviews, signs and submits the PR.
8. Establish repeated-request memory behavior with bounded diagnostics. If a
   runtime investigation blocks full validation, advance independent precision,
   API or packaging work; do not spend consecutive turns only restating
   readiness. Do not repeat device experiments solely because a log was
   truncated: page it using `port_log`.
9. Call `port_submit` when the current source passes bringup and is ready.
   This freezes it for the independent full corpus. Full acceptance is not
   guaranteed.

Only `port_write` persists edits; files written by a remote shell are
job-local. Native opencode filesystem access is read-only (edit/exec are denied
in the agent config); the separately authorized `port_write` service
enforces the allowed source paths. Use it for
permitted edits and STATE/REPORT, retaining any actual denial. CPU-only checks
use bounded `port_run` jobs on the TT host without opening a TT device; no
separate local runner is required. An empty compact status means no current user
jobs, not missing tools. Supervisors should say "no TT device
opens/evaluations/timings" when permitting CPU development, rather than
ambiguously banning every job. `port_files/read/write/run/evaluate/status/
log/submit` are your workflow tools. Each run has an exclusive card lease;
independent diagnostic jobs can use different healthy cards. Read truncated
stdout/stderr with `port_log(job_id, offset_bytes=0)` and continue using
`next_offset_bytes`; do not repeat experiments to recover logs. Keep state
within about eight minutes — a strong target, not a hard rule: overrun only for clearly
promising in-flight work, after appending a one-line progress checkpoint to STATE.md (the
runtime hard cap is thirty minutes per turn): the supervisor starts fresh bounded turns using your saved
files. Do useful work while jobs run; do not spend a turn sleeping or launching
duplicate tests. Keep mutable notes in Markdown, because code/configuration
changes invalidate earlier correctness results.

Compute must remain on TT, with real weights and input-dependent outputs.
Preserve masks, patch boundaries, scaling, row independence and repeated-call
determinism. Do not exploit the dataset, bypass tests, cache answers, or change
the contract. Optimize transferable inference behavior, including awkward
shapes and repeated calls. A good engineering result explains failures and
unsupported cases honestly; report the unsupported envelope (longer contexts,
multivariate/covariate modes) explicitly instead of implying coverage.
