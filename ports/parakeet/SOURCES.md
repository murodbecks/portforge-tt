# Start from public evidence

Read CONTRACT.md first. Target: `nvidia/parakeet-tdt-0.6b-v3` at the pinned
revision — FastConformer encoder (24 blocks, hidden 1024, 8 heads, FFN 4096
SiLU, depthwise conv kernel 9, 8× subsampling via two stride-2 convs, 128 mel
bins) + TDT transducer decoder (2 LSTM-ish layers, hidden 640, joint ReLU,
durations 0–4, max 10 symbols/step, vocab 8193). Our envelope: batch 1–4,
clips ≤32 s, greedy. Streaming/chunked inference is out of scope.

- [Parakeet TDT v3 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
  and the pinned `config.json`/`generation_config.json`: native
  `transformers` support (`ParakeetForTDT`), CC-BY-4.0 weights, 25 European
  languages. The repository also ships `.nemo`/`.gguf` variants — not pinned.
- [TDT paper (Token-and-Duration Transducer)](https://arxiv.org/abs/2309.08146)
  and [FastConformer](https://arxiv.org/abs/2005.08100): why the decode loop
  emits multiple symbols per step and how subsampling shapes encoder lengths.
- [transformers ParakeetTDT modeling source](https://github.com/huggingface/transformers/tree/main/src/transformers/models/parakeet)
  (`modeling_parakeet_tdt.py`, feature extractor, generation): the exact
  reference semantics our oracle uses — masking, greedy loop, duration rules.
  Apache-2.0; do not copy incompatible code.
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (Apache-2.0): the original
  optimized implementation — study its CUDA kernels and fuse patterns for the
  encoder conv stack; the A100 reference timings are the NVIDIA comparison
  target, matched comparisons only.
- [TT-Metal Whisper ports](https://github.com/tenstorrent/tt-metal/tree/main/models)
  (`models/demos/audio`, ttnn conv/attention ops): the closest TT precedent —
  log-mel frontends, conv subsampling and autoregressive decode loops on
  Tenstorrent. Similar architecture is a starting point, not proof of
  equivalent masking, subsampling or joint-network semantics.
- [TTNN docs](https://docs.tenstorrent.com/tt-metal/latest/ttnn/): conv ops
  (grouped/depthwise), SDPA, traces. The autoregressive TDT loop is the main
  dispatch-overhead risk — plan trace capture and host-side batching early;
  the NLLB and Chronos-2 campaigns' trace/batching wins transfer directly.

This is a shared machine (many users' home directories). Never read, write or
probe beyond this campaign's workspace and mounts; never reset cards; keep all
artifacts inside the run workspace. Precision guidance (BF16 primary, BFP8_B
as a separate block-float experiment, sensitive-op FP32 exceptions with
measured justification) matches the previous contracts. Use web search for
existing public ports and optimization write-ups; confirm claims in primary
sources; cite licenses.
