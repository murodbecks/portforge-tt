# portforge-tt — autonomous model porting to Tenstorrent

A [CHIA](https://chialoops.ai) loop that ports a pretrained model to Tenstorrent
hardware with TTNN. A **supervisor** agent plans one bounded unit of work at a
time; a **worker** agent researches the model, writes the TTNN backend, and runs
isolated jobs on a TT card through scoped MCP tools. A trusted evaluator compares
every candidate with an independent FP32 reference, first on small suites and
finally, once, on a frozen held-out corpus.

The model-specific half of a campaign is a small **port pack**; everything else is
shared. Packs for four models are included — NLLB-200 (translation), Chronos-2
(time-series forecasting), ESM-2 650M (protein language model) and Parakeet TDT
0.6B v3 (speech recognition) — together with a template for your own.

The runner, sandbox and guides target Tenstorrent Blackhole with TT-Metal. The
structure (agent loop, trusted evaluator, frozen suites, receipts) is
hardware-agnostic; to target other hardware, ask a coding agent to adapt
`runner.py` and `guides/TT_GUIDE.md`.

## Install

### Prerequisites

- Python 3.10+
- [CHIA](https://github.com/ucb-bar/chia) (the framework this runs on)
- An agent CLI: [`opencode`](https://opencode.ai) and/or
  [`claude`](https://claude.com/claude-code), logged in

### Install CHIA

```sh
# from PyPI (recommended)
pip install chialoops

# or from source
git clone https://github.com/ucb-bar/chia
pip install -e chia
```

Verify: `python -c "import chia; print('ok')"`

### Clone and test this repo

```sh
git clone https://github.com/murodbecks/portforge-tt
cd portforge-tt
pip install ray pyyaml numpy
python -m pytest tests/ -q   # 170 tests, no hardware needed
```

## How a campaign works

```
            ┌──────────── controller (portforge_loop.py, your machine) ────────────┐
            │  supervisor ──instruction──▶ worker ──MCP tools──▶ PortTools          │
            │      ▲                          │                    │                │
            │      └──── observation ─────────┘            port_run / port_evaluate  │
            └───────────────────────────────────────────────────────┼──────────────┘
                              ssh │                                  │ ssh
               ┌──────────────────▼─────────┐        ┌───────────────▼──────────────┐
               │ TT host: runner.py          │        │ reference host: evaluate.py  │
               │ bwrap sandbox, card leases, │        │ FP32 oracle (GPU or CPU),    │
               │ backend.py + candidate()    │        │ assess() → verdict           │
               └────────────────────────────┘        └──────────────────────────────┘
```

1. **Baseline** — the worker measures PyTorch on the TT host before porting.
2. **Smoke → bringup** — `port_evaluate` runs the agent's `backend.py` on TT and the
   evaluator gates it against the FP32 oracle (numerics, determinism, batching).
3. **Optimize and package** — profiling, matched same-precision comparisons, and a
   tt-metal-style package (`tt/ demo/ reference/ tests/ benchmarks/ docs/`).
4. **Submit** — `port_submit` freezes the source; the controller runs the held-out
   `full` corpus once and records `accepted_research_port` or `full_failed`.

Agents never get shell or file-write access on your machine: they read the
workspace and act only through the tools. Each run freezes the exact harness,
pack and evaluator it is graded by in `runs/<run>/harness/`.

## Layout

| Path | Contents |
|---|---|
| `portforge_loop.py` | Controller: launch, resume, supervisor/worker turns, MCP tools, submission |
| `agent.py` | Supervisor and worker nodes on the opencode or Claude Code CLI |
| `runner.py` | TT host: sandboxed jobs, card leases, deadlines, quarantine |
| `cluster.py`, `cluster.yaml` | SSH transport and the machine inventory template |
| `port.py` | Loads and freezes port packs |
| `portforge_eval.py` | Evaluator helpers shared by every pack |
| `models.py` | Pinned checkpoint registry: verify or download weights |
| `compare.py` | Compare two saved evaluation jobs under matched conditions |
| `setup_reference.sh` | Build the FP32 reference environment for a pack |
| `prompts/` | Worker and supervisor instructions |
| `guides/` | `TT_GUIDE.md` and `LESSONS.md`, given to every campaign's agents |
| `ports/<name>/` | Port packs (`template/` to start a new one) |
| `AGENTS.md` | Guide for a coding agent helping you add a model and set up machines |

A port pack holds `port.json` (manifest), `models.json` (pinned checkpoint),
`evaluate.py` (suites, reference, candidate, assess), `CONTRACT.md` and `TASK.md`
(what the agents must build), and optionally `BENCHMARK.md`, `SOURCES.md` and the
frozen `corpus.json`. The NLLB and Parakeet corpora are too large to ship here;
pass them with `--corpus`.

## Machines

You need three machines, which may overlap:

- **Controller** — where you run `portforge_loop.py` with CHIA installed, plus the
  `opencode` and/or `claude` CLI logged in.
- **TT host** — a built TT-Metal checkout, a Python environment that imports `ttnn`
  and `torch`, `bwrap`, and the model checkpoint:
  `python models.py <key> <cache-dir> --download --registry ports/<name>/models.json`.
- **Reference host** — ideally an NVIDIA GPU: copy the pack's `port.json` there and
  run `bash setup_reference.sh port.json`.

Copy `cluster.yaml`, fill every `<placeholder>`, and pass it with `--cluster`.

## Run a campaign

From this directory, with CHIA installed (`pip install chialoops`):

```sh
python portforge_loop.py --port ports/chronos2 --cluster my-cluster.yaml --hours 6 --detach
```

The command prints the run directory and returns. Progress is in
`runs/<run>/status.json`, turns in `runs/<run>/{worker,supervisor}-turns/`, and job
receipts in `runs/<run>/jobs/`. Useful options:

- `--agent claude` (default `opencode`), `--agent-model`, `--supervisor-agent`.
- `--resume runs/<run> --resume-note "..."` continues a stopped or submitted campaign
  in a new run with the same workspace; the note reaches both agents as a trusted
  operator instruction. FP32 references are reused only when the evaluator is unchanged.
- `--corpus path/to/corpus.json` for packs that do not ship one.

When the provider reports a usage limit, the controller waits for the reset if the
budget allows, and pauses otherwise.

## Add a model

Copy `ports/template/` to `ports/<name>/` and fill it in, or ask a coding agent to
do it with you by following `AGENTS.md`. In short: pin the checkpoint in
`models.json`, choose and freeze a benchmark, implement the four evaluator
functions, write the contract, run the reference smoke, then launch.

## Tests

```sh
python -m pytest tests/ -q                                        # shared harness
python -m unittest discover -s ports/<name>/tests                 # one pack's evaluator
```

No hardware, network or credentials are needed.

## Reproducing the included port results

The four included packs were produced by running this harness on Tenstorrent
Blackhole (p150) with tt-metal `fd80faa3` (Python 3.10, torch 2.14+cpu,
numpy 1.26.4) against FP32 references on an NVIDIA A100-SXM4-40GB. The
autonomous agents were `opencode` with `zai-coding-plan/glm-5.3` and
Claude Code with `claude-opus-5-5`; the agent model does not affect the
gates — a different agent (or a human implementing the same backend)
produces the same evaluation, because the evaluator, corpus and gates are
frozen per pack and each run's harness is archived in `runs/<run>/harness/`.

To reproduce a campaign end to end:

1. **Prerequisites** (see [Machines](#machines) for details):
   - a Tenstorrent Blackhole card with a built TT-Metal checkout (any recent
     commit; the included results used `fd80faa3b35fa6d38a92d08326205fc4168284ec`)
   - an NVIDIA GPU (or a strong CPU) for the FP32 reference
   - an `opencode` or `claude` CLI login on the controller

2. **Stage the checkpoint** on the TT host:
   ```sh
   python models.py chronos2 /path/to/cache --download --registry ports/chronos2/models.json
   ```

3. **Build the reference environment** (on the reference host):
   ```sh
   scp ports/chronos2/port.json reference-host:/tmp/
   ssh reference-host 'bash setup_reference.sh /tmp/port.json'
   ```

4. **Run the reference smoke** (validates the evaluator's API assumptions once):
   ```sh
   # on the reference host, with the pack's evaluate.py, models.py, models.json
   # and corpus.json copied to a working directory:
   PORTFORGE_REFERENCE_EXECUTION=direct python evaluate.py reference \
     --stage smoke --corpus corpus.json --out /tmp/ref-smoke
   ```

5. **Fill `cluster.yaml`** — copy it, replace every `<placeholder>` with your
   machine addresses, SSH keys, TT-Metal path and checkpoint path.

6. **Launch the campaign**:
   ```sh
   python portforge_loop.py \
     --port ports/chronos2 --cluster my-cluster.yaml --hours 6 --detach
   ```

7. **Observe the gates**: `runs/<run>/status.json` transitions from
   `preflight` → `reference_preflight` → `working` (turns) → `full_validation`
   → `accepted_research_port` or `full_failed`. Every evaluation job's
   request, source snapshot and verdict is archived under `runs/<run>/jobs/`
   and is independently inspectable.

To reproduce a specific gate result without an agent (e.g., to verify that
a past backend passes the same evaluation), use the archived source snapshot
and the frozen evaluator:

```sh
# from a run directory
python runs/<run>/harness/evaluate.py candidate \
  --input runs/<run>/references/<stage>/input \
  --output /tmp/recheck \
  --weights /path/to/checkpoint --precision bf16
python runs/<run>/harness/evaluate.py assess \
  --input runs/<run>/references/<stage>/input \
  --oracle runs/<run>/references/<stage>/oracle \
  --output /tmp/recheck
```

### Expected results per pack

| Pack | smoke | bringup | full | notes |
|---|---|---|---|---|
| chronos2 (bf16) | PASS (NRMSE 0.009) | PASS (NRMSE 0.043) | not run | traced p50 5.7 ms |
| esm2 (bf16) | PASS (NRMSE 0.030) | FAIL 4/8 | not run | bf16 argmax near-ties + long hidden 0.042 (documented limits) |
| nllb (bf16) | PASS | PASS (8/8) | 17/20 | BFP8_B full corpus passed |
| parakeet (bf16) | PASS (NRMSE 0.018) | PASS 8/8 (NRMSE 0.033) | 18/20 | p50 85 ms, comparable to A100 FP32 |

The exact NRMSE values depend on the TT-Metal revision and card stepping;
the pass/fail pattern is stable across revisions we tested. The two `full`
failures (nllb 3/20, parakeet 2/20) are intrinsic bf16 precision limits at
near-tie logits — both are documented with device evidence in the archived
run notes and were investigated with fp32 and mixed-precision variants
(all rejected; see the pack's `CONTRACT.md` for the full rejection table).

### Pack corpora

Chronos-2 and ESM-2 ship their frozen corpora in `ports/<name>/corpus.json`.
The NLLB corpus is derived from FLORES-200 and the Parakeet corpus from
LibriSpeech dev-clean; both are too large to ship and can be rebuilt
deterministically from their documented provenance (see
`ports/<name>/SOURCES.md`).

## License

Apache-2.0 (see [LICENSE](LICENSE)).
Model checkpoints retain their own licenses (see each pack's `models.json`).
