# Lessons from earlier campaigns (binding rules)

Distilled from four Blackhole campaigns (NLLB-200, Chronos-2, ESM-2 and
Parakeet TDT, September 2026) and a failed first NLLB attempt before them. Each
rule cost hours to days when it was broken.

## Order of work

1. **Baseline before port, port before optimization.** Measure PyTorch (and
   ONNX/forge if installed) on the fixed suites from the TT host first, pass the
   correctness gates second, optimize against the recorded baseline third. An
   optimization claim without a preserved, matched baseline artifact is invalid.
2. **Tiny tests during bring-up; the full corpus only at the end.** Use `smoke`
   and small component checks until correct, `bringup` to gate, and `submit`
   once, on frozen source. Never evaluate the full corpus during development.
3. **One bounded step per turn; checkpoint STATE.md within about eight minutes.**
   Batch routine gates on unchanged source (about four gates or six minutes),
   then write STATE.md. For a promising step still running, save a one-line
   checkpoint first and finish it, but never pass the 30-minute turn cap.
4. **Prepare submission materials early.** Build the tt-metal-style package,
   branch name, commit message and PR body as a mid-campaign milestone. The
   operator reviews, signs and submits; agents never push or open PRs.

## Experiments

5. **Never reset cards.** A timeout quarantines a card; only the controller's
   preflight recovery re-admits it. Report and continue other work.
6. **No repeat without a new hypothesis.** Rerunning an unchanged job to "see if
   it works now" wastes device hours. For hangs, write a small timed reproducer.
7. **Check the reference before trusting a failing probe.** A Chronos-2 worker
   spent seven probe generations on "GEMM poisoning" that was really its own
   test comparing `x @ W` on the host with a device holding `W.T`. When a
   control that should pass fails, suspect the probe first.
8. **Truncated logs are not a reason to rerun.** Page saved logs with
   `port_log(job_id, offset_bytes=…)` and continue from `next_offset_bytes`.
9. **Poll pending jobs; never duplicate them.** Use `port_status` and do
   independent work while jobs run.
10. **Files change only through `port_write`.** Edits made inside a
    `port_run` job are discarded when the job ends; each job starts from the
    written workspace.

## Evidence

11. **Historical is not current.** A pass for an older source, cluster or suite
    is context, never a current gate. Adding any non-Markdown file changes the
    source identity, so gates must be rerun on the final source before submit.
12. **Record what did not work.** Rejected hypotheses, with measurements, were
    as valuable to reviewers as the wins.
13. **Gates match the task and are never widened.** Token-exact decoding,
    float NRMSE and task metrics (WER, chrF, weighted quantile loss) all appear
    in earlier packs. If a decision flips, measure its margin: a near-tie in
    the model itself is a documented limit, not a reason to change a gate.
14. **Honest envelopes.** Inputs beyond the frozen envelope and features the
    contract excludes are reported as unsupported, not implied.
15. **Telling agents about held-out failures makes those cases non-blind.**
    Operators may share a failing full-corpus case in general terms; the report
    must then say that result is no longer a blind held-out measurement.

## Workspace and machines

16. **Keep the workspace small.** No recursive note archives, no embedded logs
    in STATE.md, no copies of old runs.
17. **Shared-machine discipline.** Touch nothing outside the campaign's
    workspace and mounts, keep host load bounded, and never reset cards.
18. **Package in tt-metal style from the start.** Follow existing model layouts
    (`tt/ demo/ reference/ tests/ benchmarks/ docs/`, SPDX headers, a README
    with measured evidence). A late layout change invalidated benchmark
    provenance once.
