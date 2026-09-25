# Fresh NLLB port

Build one correct, fast, configurable TTNN implementation for NLLB distilled
600M, distilled 1.3B and 3.3B.
Read CONTRACT.md for the exact interface and acceptance envelope. Use public
research and your own implementation; earlier private ports are unavailable.
Start with the real 600M checkpoint unless the campaign selects another model.
That selection is the development anchor; it does not remove the other sizes
from the shared-artifact goal. Smaller layer experiments are diagnostics,
never a substitute for full-model correctness.

1. Read TT_GUIDE.md and SOURCES.md; inspect public implementations and pinned installed source.
   Save concise findings and citations in RESEARCH.md. Explain model structure,
   generation semantics and precision choices before implementing.
2. Write backend.py and useful component checks. Use `port_run` for isolated
   experiments and source inspection. Python, PyTorch and TTNN already exist;
   the installed TT-Metal tree is `/opt/tt-metal`. The selected checkpoint is
   mounted under `/weights`; 600M uses `/weights/pytorch_model.bin`, while larger
   models use checkpoint directories. Do not install packages or reset devices.
3. Run `port_evaluate(stage="smoke")` for one short real-model request. An
   independent FP32 PyTorch reference runs on the configured NVIDIA compute host
   (Slurm allocation or dedicated GPU VM; never a login node).
   Use `port_status(compact=True)` to collect asynchronous results; retain job IDs. Inspect
   the failed phase, fix one hypothesis, and avoid duplicate pending jobs.
4. Pass `bringup` before optimizing. Preserve that source and job as the baseline.
   Profile where time and memory go, investigate public optimization techniques,
   then compare changes on identical workloads and settings. Seek substantial
   measured gains; there is no required speedup target to game. Keep useful
   improvements even if the gain is smaller than hoped.
   Start with explicit BF16 candidate precision and compare FP32 when diagnosing
   numerical failures. Investigate FP16 only if this installed TTNN supports it;
   TT BFP8_B is a separate block-float experiment, not a promise of IEEE FP8.
   Declare precision exceptions and accumulation policy. Compare supported modes
   on the same requests against the unchanged FP32 oracle, and preserve a passing
   baseline per precision. Never disguise a precision change as a same-precision
   implementation speedup. CUDA FP16/BF16 benchmarks provide additional context.
5. Write REPORT.md with correctness evidence, reproducible commands, baseline
   versus optimized latency and memory, limitations and sources. Report bandwidth
   only if actually measured or clearly label an estimate. Keep STATE.md short:
   completed work, current hypothesis, pending job IDs, next action.
6. Produce a reusable artifact: validate public API inputs and checkpoint/config
   compatibility; provide a standalone text translation CLI with explicit checkpoint,
   source/target languages, device and generation cap. Tests must import actual
   code and accept configurable fixtures rather than execute Markdown or require
   CHIA's `/input` and `/weights`. Keep one configuration-driven backend.
   After a bounded optimization experiment, test that shared source on all three
   sizes: smoke first, then bringup for passing cells, with explicit `model` and
   `precision` arguments. Cover BF16 and investigate BFP8_B separately. Fix actual
   failures before more profiling of only the development model. Report missing
   or unsupported cells; a pass for one size or mode is never inherited by another.
7. Establish repeated-request memory behavior with bounded diagnostics. If a
   runtime investigation blocks full validation, advance independent precision,
   API or packaging work; do not spend consecutive turns only restating readiness.
   Do not repeat device experiments solely because a log was truncated: page it
   using `port_log`. Report unresolved failures and unsupported behavior explicitly.
8. Call `port_submit` when the current source passes bringup and is ready. This
   freezes it for the independent full corpus. Full acceptance is not guaranteed.

Only `port_write` persists edits; files written by a remote shell are job-local.
Native Codex filesystem access is read-only; the separately authorized
`port_write` service enforces the allowed source paths. Use it for permitted
edits and STATE/REPORT, retaining any actual denial. CPU-only checks use bounded
`port_run` jobs on the TT host without opening a TT device; no separate local
runner is required. An empty compact status means no current user jobs, not
missing tools. Supervisors should say "no TT device opens/evaluations/timings"
when permitting CPU development, rather than ambiguously banning every job.
`port_files/read/write/run/evaluate/status/log/submit` are your workflow tools.
`port_run(..., model="600m")` can select a registered checkpoint for standalone
checks or profiling; omitted/empty model uses the development anchor. With
`public_input_stage="bringup"`, its public inputs match the selected model.
Use this explicit routing for larger-model benchmarks; do not infer model support
from an anchor-only run. A diagnostic run never awards an evaluation pass.
Each run has an exclusive card lease; independent diagnostic jobs can use different
healthy cards. Read truncated stdout/stderr with `port_log(job_id, offset_bytes=0)`
and continue using `next_offset_bytes`; do not repeat experiments to recover logs.
Keep state within eight minutes: the supervisor starts fresh
bounded turns using your saved files. Do useful work while jobs run; do not spend
a turn sleeping or launching duplicate tests. Keep mutable notes in Markdown,
because code/configuration changes invalidate earlier correctness results.

Compute must remain on TT, with real weights and input-dependent outputs.
Preserve masks, positions, cache lifecycle, greedy stopping and batch independence.
Do not exploit the dataset, bypass tests, cache answers, or change the contract.
Optimize transferable inference behavior, including awkward shapes and repeated
calls. A good engineering result explains failures and unsupported cases honestly.
