# Start from public evidence

Read CONTRACT.md first. Target is `amazon/chronos-2` at the pinned revision:
encoder-only T5-style backbone (12 layers, d_model 768, 12 heads, d_kv 64, ReLU
FFN 3072), input patches of 16 with stride 16, RoPE (theta 10000), a register
token, arcsinh input scaling, and a direct multi-patch quantile head (21
quantiles, up to 64 output patches). Our required envelope is deliberately
smaller than the model's maximum (context ≤512, horizon ≤64, batch 1–4,
univariate); longer contexts and multivariate/covariate group attention are
explicit out-of-scope extensions. Establish the FP32 package reference and a
correct minimal TTNN implementation before optimization. Keep a short
RESEARCH.md with source URL, revision/date, relevant finding, and the
experiment it motivates. Documentation describes possibilities; installed
source and measured tests decide what actually works on this Blackhole runtime.

In the isolated runner, `/opt/tt-metal` exposes the installed public source but
hides Git metadata. `git rev-parse` there is expected to fail; use the
controller's verified runtime identity and saved receipts. That failure alone
is not a device or model failure. Keep ordinary package imports and pytest
collection portable; passing tests from the model's own directory does not
prove upstream collection. The checkpoint mount contains
`config.json` + `model.safetensors`; inspect `/weights` rather than assuming a
layout.

- [Chronos-2 technical report](https://arxiv.org/abs/2510.15821) and the
  [chronos-forecasting repository](https://github.com/amazon-science/chronos-forecasting)
  (`src/chronos/chronos2/{pipeline,model,layers,preprocess}.py`): patching,
  masking/missing-value semantics, scaling, RoPE, quantile head and the exact
  `predict` API. Apache-2.0; cite it and do not copy incompatible code.
- [TT-Forge model bring-up guide](https://docs.tenstorrent.com/tt-forge/model-bring-up-guide.html):
  architecture, compiler/runtime distinction, initial correctness and profiling.
  TT-Forge compiler guidance is not a promise that the same API exists in TTNN.
- [TT-Metal documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/):
  tensor layouts, supported operations and device/memory APIs. Inspect installed
  public `ttnn`/tests source for the pinned runtime before copying examples.
- [Public TT-Metal models](https://github.com/tenstorrent/tt-metal/tree/main/models):
  search T5, BERT and encoder-only demos for attention, masking, weight
  placement and tests; the community time-series bring-ups —
  [time_series_transformer PR #51875](https://github.com/tenstorrent/tt-metal/pull/51875)
  and the PatchTST/TinyTimeMixer bounty issues #32139/#32140/#32142 — are the
  closest layout and domain precedent. Similar architecture is a starting point,
  not proof of equivalent patching, positions, normalization or mask semantics.
- [TT-Metal issues](https://github.com/tenstorrent/tt-metal/issues) and
  [pull requests](https://github.com/tenstorrent/tt-metal/pulls): search
  chronos, patching/RoPE/encoder keywords and Blackhole. Record whether a change
  is merged, its tested hardware and whether it exists in the installed revision.
- Forecasting context (NOT the port gate): [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval)
  and [fev-bench](https://huggingface.co/spaces/autogluon/fev-leaderboard)
  leaderboards where Chronos-2 is the published zero-shot leader; our frozen
  ETTh1/synthetic corpus is only a bounded correctness/latency sample.

Use web search for existing public ports and relevant optimization papers, also
from other hardware. Confirm technical claims in primary sources. Cite and
preserve licenses/attribution for reused code; do not copy incompatible code or
private earlier ports. Model weights and inference package are Apache-2.0.

During bring-up use the smallest stage and isolated component tests (a single
attention block, patch embedding, quantile head). Log progress before expensive
operations and use short, explicit timeouts. Investigate shape, precision,
layout and lifecycle errors with one hypothesis at a time. Once correct, profile
synchronized end-to-end latency, cold setup separately, memory and transfers.
Use evidence to choose fusion, layout/sharding, reuse, trace capture or dispatch
reductions; the NLLB campaign's largest TT-side win came from trace capture plus
request batching on static shapes, which this fixed-envelope forecasting
workload is well suited for. Keep precision/behavior fixed for comparisons
unless an explicit quality tradeoff is reported. Test masked inputs, batching
and boundary contexts (64/65, 128, 512) before broader checks. Never skip
learned computation, cache expected answers, or optimize only named benchmark
inputs. Record regressions and rejected optimizations as well as gains.

For precision experiments, inspect [TTNN tensor formats](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tensor.html)
and the installed operator support before choosing a policy. The documentation
lists BF16, FP32 and BFP8_B; BFP8_B shares an exponent across groups of values
and can lose small values when magnitudes differ — risky for the arcsinh-scaled
patch projections and quantile head, so measure before adopting. This is not
IEEE E4M3/E5M2 FP8. Do not infer native FP16 support from an FP16 PyTorch tensor
that TTNN converts. Storage dtype, math fidelity, accumulation and
sensitive-operation exceptions are separate decisions; record each and test
actual runtime behavior.

[PyTorch numerical-accuracy guidance](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
explains overflow and reduced-precision reductions. Our diagnostic reference
benchmark loads an explicit model dtype, disables TF32 and reduced-precision
reduction settings, and compares to the frozen FP32 oracle; it does not
blanket-autocast operations. Failures are useful evidence to motivate scoped
FP32 exceptions (scaling, normalization, quantile accumulation are candidates),
not permission to increase tolerances.
