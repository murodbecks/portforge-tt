# Start from public evidence

Read CONTRACT.md first. Start with NLLB-200 distilled 600M. Establish the PyTorch
reference and a correct minimal accelerator implementation before optimization;
larger checkpoints follow a stable small-model result. Keep a short RESEARCH.md
with source URL, revision/date, relevant finding, and the experiment it motivates.
Documentation describes possibilities; installed source and measured tests decide
what actually works on this Blackhole runtime.

In the isolated runner, `/opt/tt-metal` exposes the installed public source but
hides Git metadata. `git rev-parse` there is expected to fail; use the controller's
verified runtime identity and saved receipts. That failure alone is not a device
or model failure. Keep ordinary package imports and pytest collection portable;
passing tests from the model's own directory does not prove upstream collection.
Checkpoint mounts may contain one weight file or a complete directory: inspect
`/weights` rather than assuming `/weights/config.json` exists. Request the selected
model's `public_input_stage` for its verified public config at `/input/config.json`.

- [TT-Forge model bring-up guide](https://docs.tenstorrent.com/tt-forge/model-bring-up-guide.html):
  architecture, compiler/runtime distinction, initial correctness and profiling.
  TT-Forge compiler guidance is not a promise that the same API exists in TTNN.
- [TT-Metal documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/):
  tensor layouts, supported operations and device/memory APIs. Inspect installed
  public `ttnn`/tests source for the pinned runtime before copying current examples.
- [Public TT-Metal models](https://github.com/tenstorrent/tt-metal/tree/main/models):
  search T5, Whisper, BART and SeamlessM4T for encoder/decoder attention, masking,
  weight placement, generation and tests. Similar architecture is a starting point,
  not proof of equivalent positions, normalization, cache semantics or token rules.
- [TT-Metal issues](https://github.com/tenstorrent/tt-metal/issues) and
  [pull requests](https://github.com/tenstorrent/tt-metal/pulls): search NLLB,
  SeamlessM4T, relevant operation names and Blackhole. Record whether a change is
  merged, its tested hardware and whether it exists in the installed revision.
- [Hugging Face NLLB docs](https://huggingface.co/docs/transformers/model_doc/nllb)
  and the checkpoint's configuration/model card: verify special tokens, positions,
  encoder-decoder masks, generation settings and model license.
- [CTranslate2 Transformers guide](https://opennmt.net/CTranslate2/guides/transformers.html)
  and [CTranslate2 source/tests](https://github.com/OpenNMT/CTranslate2): investigate
  mature translation inference, batching and cache techniques. Its C++ tests do
  not validate a TT implementation without adapting the behavioral contract.

Use web search for existing public ports and relevant optimization papers, also
from other hardware. Confirm technical claims in primary sources. Cite and
preserve licenses/attribution for reused code; do not copy incompatible code or
private earlier ports. Keep model-weight licensing distinct from code licensing.

During bring-up use the smallest stage and isolated component tests. Log progress
before expensive operations and use short, explicit timeouts. Investigate shape,
precision, layout and lifecycle errors with one hypothesis at a time. Once correct,
profile synchronized end-to-end latency, cold setup separately, memory and transfers.
Use evidence to choose fusion, layout/sharding, reuse or dispatch reductions. Keep
precision/behavior fixed for comparisons unless an explicit quality tradeoff is
reported. Test padding, batching and long sequences before broad quality checks.
Never skip learned computation, cache expected answers, or optimize only named
benchmark inputs. Record regressions and rejected optimizations as well as gains.

For precision experiments, inspect [TTNN tensor formats](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tensor.html)
and the installed operator support before choosing a policy. The documentation
lists BF16, FP32 and BFP8_B; BFP8_B shares an exponent across groups of values and
can lose small values when magnitudes differ. This is not IEEE E4M3/E5M2 FP8.
Do not infer native FP16 support from an FP16 PyTorch tensor that TTNN converts.
Storage dtype, math fidelity, accumulation and sensitive-operation exceptions
are separate decisions; record each and test actual runtime behavior.

[PyTorch numerical-accuracy guidance](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
explains overflow and reduced-precision reductions, including the smaller dynamic
range of FP16. [CUDA BF16 support](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html)
must be checked on the actual GPU. Our diagnostic reference benchmark loads an
explicit model dtype, disables TF32 and reduced-precision reduction settings,
and compares to the frozen FP32 oracle; it does not blanket-autocast operations.
Failures are useful evidence to motivate scoped FP32 exceptions, not permission
to increase tolerances. Search additional primary documentation for genuine FP8
support before proposing it; unsupported formats remain unimplemented experiments.
