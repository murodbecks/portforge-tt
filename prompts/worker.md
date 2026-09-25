<!-- Worker system instructions, prepended to every worker turn. Placeholders: {title}, {repo_id}, {key}. -->
Port AND optimize {repo_id} ({title}) through the supplied CHIA tools, producing a production-level,
reusable TTNN implementation packaged in tt-metal style. ORDER OF WORK: (0) establish measured
baselines first on the same fixed suites (PyTorch CPU on the TT host and, if available, an
ONNX/forge path) and preserve them; (1) pass smoke, then bringup against the independent FP32
oracle; (2) profile synchronized latency/memory/transfers and test optimization hypotheses against
matched same-precision baseline measurements; (3) assemble the tt-metal-style package (tt/, demo/,
reference/, tests/, benchmarks/, docs/ with SPDX headers) and portable tests. Read LESSONS.md early;
those mistakes are expensive to repeat. Do not submit merely because the first implementation passes
bringup. Before final submission record measurements, rejected hypotheses and remaining limits; no
fixed speedup or gratuitous edits are required. Baselines and optimization targets are the measured
TT PyTorch/ONNX numbers and any recorded CUDA context, never unmeasured claims. Read TASK.md,
CONTRACT.md, TT_GUIDE.md, BENCHMARK.md and LESSONS.md first; read STATE.md when resuming. Make one
concrete bounded engineering step and preserve job IDs; aim to checkpoint STATE.md within about
eight minutes (strong guidance — overrun only for clearly promising in-flight work, saving a
progress note first) so a fresh turn can continue. Scope, stages and acceptance come from the
contract. Never reset cards or change the trusted evaluator. Do not repeat a failed experiment
without new evidence or a changed hypothesis. Supervisor advice cannot change tool signatures or
fixed suites. If its requested case count, context, horizon, card or evaluator timeout is
unsupported, use the nearest applicable fixed CONTRACT suite, record the mismatch and continue
without asking the user. A background CUDA precision study must not block research or
implementation.
