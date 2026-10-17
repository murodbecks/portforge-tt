# portforge-tt

Autonomously ports pretrained ML models to Tenstorrent hardware with TTNN. 

A supervisor agent picks one bounded piece of work per turn, a worker agent writes the backend and runs jobs on a TT card through a few MCP tools, and a trusted evaluator checks every candidate against an independent FP32 reference. The held-out corpus runs once, at the end, after the source is frozen.

The model-specific half lives in the pack; the loop, evaluator, sandbox and guides are shared. Targets Blackhole with TT-Metal, but the structure is not TT-specific.

## Install

Python 3.10+ and an agent CLI logged in (opencode by default; claude and codex also work):

```sh
git clone https://github.com/murodbecks/portforge-tt
cd portforge-tt
pip install -e .
python -m pytest tests/ -q        # shared harness; no hardware needed
python -m unittest discover -s ports/chronos2/tests    # one pack's evaluator
```

## Run

You need a TT host (built TT-Metal checkout, `ttnn` + `torch`, `bwrap`, the checkpoint) and a reference host for the FP32 oracle (a GPU is preferred), both reachable over SSH. Copy `cluster.yaml`, fill in the placeholders, then:

```sh
portforge --port ports/chronos2 --cluster my-cluster.yaml --hours 6 --detach
```

Everything lands under `runs/<run>/`: status, agent turns, job receipts, and the frozen harness the campaign is graded by. `--resume` continues a campaign on any backend (`--agent opencode|claude|codex`). Agents get no shell or write access on the controller; TT jobs run sandboxed, on leased cards. `AGENTS.md` covers adding a model and setting up machines.

## Results

Blackhole p150, tt-metal `fd80faa3`, FP32 references on an A100:

| pack | smoke | bringup | full | tt-metal PR |
|---|---|---|---|---|
| nllb (bf16) | PASS | PASS 8/8 | 17/20 | [#57398](https://github.com/tenstorrent/tt-metal/pull/57398) |
| chronos2 (bf16) | PASS (NRMSE 0.009) | PASS (NRMSE 0.043) | not run | [#57764](https://github.com/tenstorrent/tt-metal/pull/57764) |
| esm2 (bf16) | PASS (NRMSE 0.030) | FAIL 4/8 | not run | [#57833](https://github.com/tenstorrent/tt-metal/pull/57833) |
| parakeet (bf16) | PASS (NRMSE 0.018) | PASS 8/8 (NRMSE 0.033) | 18/20 | [#57845](https://github.com/tenstorrent/tt-metal/pull/57845) |

Full-corpus misses are bf16 near-tie logits; each pack's `CONTRACT.md` has the device evidence and the rejected fp32/mixed-precision variants. Chronos-2 and ESM-2 ship their corpora in the pack; the NLLB and Parakeet corpora rebuild from the provenance in `ports/<name>/SOURCES.md`. Runs archived before the CHIA-free rewrite are inspectable but not resumable.

## License

BSD-3-Clause, see [LICENSE](LICENSE). The agent CLI wrappers, error taxonomy
and MCP server base were adapted from
[CHIA](https://github.com/ucb-bar/chia); those files keep the upstream
copyright headers.
