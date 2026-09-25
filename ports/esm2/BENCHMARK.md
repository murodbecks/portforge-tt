# Benchmark: what the port is measured on, and why

A **porting benchmark**, not a biology-leaderboard claim. Gates answer: (1) is
the TT implementation numerically faithful to the FP32 oracle (logits +
final hidden), (2) does it preserve batch/padding/revisit semantics, (3) is it
faster relative to its own recorded baseline, with A100 timings as context.

## Frozen corpus (`corpus.json`, sha256 `369c475f…`)

| Sequences | Origin | Why |
|---|---|---|
| 14 Swiss-Prot proteins (68–1273 aa; spike truncated to 1024) | [UniProtKB](https://rest.uniprot.org), CC-BY-4.0, deterministic accessions | Real, reviewed proteins spanning lengths/j families: enzymes, antibodies, cytoskeleton, signal proteins; length spread exercises padding and rotary wrap cases. |
| syn-max1024 / syn-1023 / syn-single / syn-pair / syn-unknown-X | deterministic | Envelope boundary (1023/1024 residues), minimal sequences (1, 2), unusual residues (X/Z/B → unk). |

Deterministic masking: residue positions 5, 21, 37, … are replaced by
mask_token 32 by the evaluator; candidates see only already-masked ids.

## Stages and gates

- `smoke` 1 case; `bringup` 8; `full` 21 (all sequences + batch4/reverse/
  revisit/single-residue; 3 timed repeats).
- logits + hidden max-row NRMSE ≤0.04 vs FP32 oracle; masked-position argmax
  exact; behavior NRMSE ≤0.01; repeat determinism.
- Diagnostics: residues/second, first-call latency, memory snapshots.

## Why not big biology benchmarks

PERM/ ProteinGym mutation-effect or TAPE measure model quality across
thousands of variants; they are post-port validation, not porting gates.
Coverage beyond 1024 residues (ESM-2's own limit), structure heads and
fine-tuning are explicit out-of-scope extensions.
