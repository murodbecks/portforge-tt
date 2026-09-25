# ESM-2 650M → Blackhole port

Build one correct, fast, configurable TTNN implementation for
`facebook/esm2_t33_650M_UR50D` (652M, 33-layer masked-LM protein encoder, MIT),
packaged in tt-metal style for an upstream `tt-metal` PR. Read CONTRACT.md,
BENCHMARK.md and LESSONS.md first. This is a shared machine — never touch other
users' files, never reset cards; keep everything inside this campaign's
workspace.

1. RESEARCH: rotary position embeddings, token dropout, contact/attention
   structure, the MLM head — write RESEARCH.md with citations before
   implementing.
2. BASELINE FIRST: PyTorch FP32 baseline on the TT host; A100 FP32/BF16
   reference timings are the comparison context. Optimization claims cite
   preserved matched baselines only.
3. backend.py via `port_run` experiments; pinned checkpoint under `/weights`;
   tokenization is deterministic CPU policy.
4. `port_evaluate(stage="smoke")` → FP32 oracle check → `bringup` → profile →
   optimize (BF16 anchor; FP32 diagnosis; BFP8_B separate) → re-gate →
   `port_submit`.
5. tt-metal-style deliverable (tt/ demo/ reference/ tests/ benchmarks/ docs/,
   SPDX headers); CLI: FASTA in → per-residue embeddings/logits out. The
   operator reviews, DCO-signs and submits the PR.

Only `port_write` persists edits. `port_run/evaluate/status/log/submit` are the
workflow tools; page truncated logs, never rerun. Checkpoint within ~8 minutes
(strong target; hard cap 30). Compute on TT with real weights; preserve
padding masks, batch independence and determinism. Sequences beyond 1024
residues, fold/structure heads (ESMFold) and fine-tuning are out of scope,
reported honestly.
