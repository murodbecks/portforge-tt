# Start from public evidence

Read CONTRACT.md first. Target: `facebook/esm2_t33_650M_UR50D` at the pinned
revision — 33-layer BERT-style encoder, hidden 1280, 20 heads, FFN 5120 GELU,
rotary positions, vocab 33, token dropout, masked-LM head. Envelope: batch
1–4, ≤1024 residues, single forward (no decode loop — the easiest of our
ports; leverage that for trace capture and batching wins early).

- [ESM-2 paper (Lin et al., Science 2023)](https://www.science.org/doi/10.1126/science.ade2574)
  and [facebookresearch/esm](https://github.com/facebookresearch/esm) (MIT):
  evolution-scale models, rotary choice, contact prediction usage.
- [transformers ESM modeling source](https://github.com/huggingface/transformers/tree/main/src/transformers/models/esm):
  the exact reference semantics our oracle uses (rotary application, token
  dropout, EsmForMaskedLM head). Apache-2.0.
- [TT-Metal BERT ports](https://github.com/tenstorrent/tt-metal/tree/main/models)
  (`bert`, `bert_large_perf`, `distilbert`, `modernbert`): the closest TT
  precedent for encoder blocks, rotary/layernorm handling and trace-based
  inference; similarity is a starting point, not proof of identical rotary or
  dropout semantics.
- TTNN docs for SDPA/rotary/layout ops. Precision guidance unchanged (BF16
  anchor; BFP8_B separate; declared FP32 exceptions with measured
  justification). Use web search for prior ESM ports; confirm claims in
  primary sources; cite licenses.

Shared machine: never touch other users' files; never reset
cards; keep artifacts inside the workspace. Another user keeps idle tmux
sessions here — be a polite neighbor.
