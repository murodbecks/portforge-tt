# <Model> benchmark

<!-- Provenance and rationale for corpus.json. Agents read this; reviewers audit it. -->

## Frozen corpus (`corpus.json`)

TODO: what the corpus contains, where each item comes from (URL, revision or
access date, license), and how items were selected. Include deterministic
synthetic edge cases (silence, constants, extreme dynamic range, minimum and
maximum lengths) with their generator and seed.

The corpus is a JSON object with `schema_version: 1`, a `sources` list
(provenance and licenses), `cases` (one entry per clip, keyed by name, holding
the input data and an optional `label`), and `selection` naming the clips used
by each stage (`smoke`, `bringup`, `full`). Its identity is the SHA-256 of its
canonical JSON; pin that in `port.json` and `evaluate.py`.

## Gates and why

TODO: each gate in `GATES`, what it protects against, and why the threshold is
appropriate for this task.

## Held-out discipline

`full` cases are never shown to agents during development. If an operator
shares a failing full case, later full results are no longer blind for it.
