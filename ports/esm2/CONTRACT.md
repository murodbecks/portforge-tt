# ESM-2 backend contract

Write `backend.py` from scratch for pinned `facebook/esm2_t33_650M_UR50D`
(model key `esm2`, MIT). Configuration values, not hardcoded dimensions; the
registry in `models.json` is authoritative. The `.bin`/`.h5` repository
duplicates are not part of the pinned manifest.

```python
def create_backend(weights_path, config, device, *, precision="bf16"): ...

class Backend:
    precision_policy = {"mode": "bf16", "weights": "bf16", "activations": "bf16",
                        "accumulation": "fp32", "exceptions": []}

    def embed(self, input_ids, attention_mask):
        # CPU numpy int64 [B,L] (already-masked: masked positions carry
        # mask_token_id 32) and int64 [B,L]. Return CPU numpy float32:
        # {"logits": [B,L,33], "hidden": [B,L,1280]} — the MLM head output and
        # the final encoder hidden states. Never mutate inputs; deterministic
        # across repeated identical calls.
```

Tokenization and the frozen masking rule (positions 5, 21, 37, … residue
index +1 for the cls token) are host CPU policy; learned inference on TT.
Precision semantics, `DEVICE_OPTIONS`, receipts/freeze/anti-cheat rules match
the previous campaigns. The immutable oracle is FP32
(`transformers.EsmForMaskedLM`) on the reference host.

Gates: logits and final-hidden max-row NRMSE ≤0.04 vs the oracle on every
numerical case; masked-position argmax exactly matches the FP32 oracle rows;
batch/single/reorder/revisit hidden rows agree within NRMSE ≤0.01; repeated
calls bit-identical. Envelope: batch 1–4, sequences ≤1024 residues (pad to
≤1026 tokens). Timing protocol:
`token_ids_to_logits_with_sync_excluding_load_tokenization_progress_v1`.
Baselines before optimization claims; TT-vs-TT ratios never mix with A100
context.

Stages: `smoke` = 1 sequence; `bringup` = 8 (batch 2 + single/reorder, 1024
max-length, 2-residue, unknown-residue X/Z/B cases); `full` = all 14 Swiss-Prot
sequences + batch4 + reverse + revisit + single-residue, three timed repeats.

## tt-metal deliverable

`models/experimental/esm2` in tt-metal style with SPDX headers and README
(measured evidence; MIT applies to weights and code). Standalone CLI: FASTA in
→ embeddings/logits out with explicit checkpoint/device/precision arguments.
The worker prepares branch/commit/PR body; the human operator submits.
