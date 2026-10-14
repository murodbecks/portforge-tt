# Adding a model to portforge-tt (guide for coding agents)

You are helping a user port a new model to Tenstorrent with this harness. Work
with them step by step; the decisions marked **Ask** belong to the user. Never
run a campaign, touch remote machines, or spend GPU/TT time without their
go-ahead. The finished result is a new port pack in `ports/<name>/`.

Read `README.md` first, then the closest existing pack:

| Model shape | Closest pack |
|---|---|
| Encoder, float outputs | `ports/esm2` (logits), `ports/chronos2` (quantile forecasts) |
| Encoder-decoder or autoregressive tokens | `ports/nllb` (translation), `ports/parakeet` (speech) |

## 1. Choose the model and pin it

**Ask**: the Hugging Face repo and license, which checkpoint sizes to deliver, and
the envelope (batch sizes, input lengths, output lengths, languages or modes).

- Copy `ports/template/` to `ports/<name>/`.
- In `models.json`, set `repo_id`, an immutable `revision` (40-character commit),
  the weight files and their SHA-256 and byte sizes (from the Hugging Face LFS
  API, not from memory), and the `config.json` fields the port relies on under
  `architecture`.
- Check it: `python models.py <key> /tmp/ckpt --download --registry ports/<name>/models.json`.

## 2. Choose the benchmark with the user

**Ask**: which public data represents real use, its license, and which edge
cases matter (empty or constant inputs, extreme values, shortest and longest
lengths, masking).

- Build `corpus.json` (schema in `ports/template/BENCHMARK.md`): real items from
  the chosen data plus seeded synthetic edge cases, and a `selection` for
  `smoke` (1 case), `bringup` (about 8) and `full` (held out).
- Record provenance, licenses, generators and seeds in `BENCHMARK.md`.
- Compute its identity and pin it in both `port.json` and `evaluate.py`:
  `python -c "import json, cluster; print(cluster.identity(json.load(open('ports/<name>/corpus.json'))))"`.
- Keep `full` cases out of development; agents never see them.

**Ask**: the gates. Earlier packs use max-row NRMSE ≤ 0.04 against the FP32 oracle,
≤ 0.01 between batch and single-row runs, bit-identical repeated calls, exact
tokens for decoders, and a task metric on `full` (for example ≤ 2% worse WER).

## 3. Write the evaluator

Implement the four functions in `ports/<name>/evaluate.py`:

- `make_suite(corpus, stage, model)` builds the three suites and returns
  `(spec, labels)`. Every `compare_to` must name a case of the same stage;
  `portforge_eval.check_suite` enforces this and the launcher rejects bad suites.
- `reference(args)` runs the official implementation in FP32 on the reference
  host and writes public inputs and private oracle outputs. Disable TF32 for
  matmuls and cuDNN convolutions.
- `candidate(args)` is generic; fill in `forward_case()` to call the backend
  method named in `CONTRACT.md`.
- `assess(args)` compares actual and oracle outputs and writes the verdict.

Add `ports/<name>/tests/test_evaluate.py` (see an existing pack) and run it with
`python -m unittest discover -s ports/<name>/tests`.

## 4. Write the agent documents

- `CONTRACT.md`: the backend interface, envelope, gates, and deliverable.
- `TASK.md`: the ordered plan; keep the template's order.
- `SOURCES.md`: model card, papers, reference code, closest tt-metal ports.
- `port.json`: title, task, delivery models, precisions, corpus hash, and the
  reference host's packages and import checks.

Run `PYTHONPATH=.:../..:tests python -m unittest tests.test_ports`: it loads
every pack and checks hashes, suites, tool names and site-specific leaks.

## 5. Set up the machines with the user

**Ask**: which TT host and which reference host (GPU preferred) to use, SSH users and
keys, and which cards the campaign may lease.

- TT host: a built TT-Metal checkout, its commit, a Python environment with
  `ttnn`, `torch`, `numpy`, `pytest`, `bwrap`, and the checkpoint downloaded with
  `models.py --download`. List the cards with `ls /dev/tenstorrent` or `tt-smi`.
- Reference host: copy `port.json` and run `bash setup_reference.sh port.json`.
- Fill a copy of `cluster.yaml`; keep it out of version control.
- Reference smoke on the reference host, once, before any campaign:
  `PORTFORGE_REFERENCE_EXECUTION=direct python evaluate.py reference --stage smoke --corpus corpus.json --out /tmp/ref-smoke`
  (with `evaluate.py`, `portforge_eval.py`, `models.py` and `models.json` copied beside it).

## 6. Launch and supervise

**Ask**: the agent backend (`--agent opencode` or `claude`), budget hours, and
whether to watch the run together.

```sh
python portforge_loop.py --port ports/<name> --cluster my-cluster.yaml --hours 6 --detach
```

While it runs, read `runs/<run>/status.json` and the turn files. To steer the
agents, stop the controller and `--resume` with a `--resume-note`; editing the
workspace's STATE.md does not reach the supervisor as a trusted instruction. If
you share a failing held-out case with the agents, record that its later result
is no longer blind.


## 7. Upstream tt-metal integration (after the campaign)

When a model package is ready for an upstream PR to `tenstorrent/tt-metal`:

- **Branch**: create `murobecks/<model>-blackhole` from `origin/main` (never from
  another model's branch). One model per branch, no prior commits.
- **Layout**: place the package under `models/experimental/<model>/` following the
  existing layout (`tt/ demo/ reference/ tests/ benchmarks/ docs/ README.md`).
- **Tests**: convert campaign tests to pytest (`def test_*` with assertions, not
  `main()` scripts). Include at least: portable CPU tests (no device needed),
  on-device bring-up tests (random weights, small config), and one end-to-end
  test with the real checkpoint asserting NRMSE/argmax against the FP32 reference.
- **Checkpoint paths**: resolve via env var (`<MODEL>_WEIGHTS`) or the CI HF cache
  (`/mnt/MLPerf/huggingface/hub/models--<org>--<name>/snapshots/<rev>`). Never
  hardcode `/weights`.
- **NRMSE gates**: set with ~30% headroom over the worst measured value across
  multiple runs — a gate tuned exactly to one measurement will flake on CI.
- **Formatting**: run `black --target-version py310` and `isort --profile black`
  before pushing (tt-metal pre-commit hooks).
- **Docs**: remove references to development probes (`probe_*.py`, `diag_*.py`)
  that are not in the PR.
- **CI entry**: after review approval, add the Tier 3 unit-test entry to
  `tests/pipeline_reorg/models_unit_tests.yaml`:

```yaml
- name: <model> unit tests
  cmd: |
    pytest models/experimental/<model>/tests --timeout=300
  model: <model-name>
  model_family: <Family>
  skus:
    bh_p150:
      timeout: 5
      tier: 3
  owner_id: <YOUR_SLACK_MEMBER_ID>  # ask the user for their Slack member ID
  team: models
  budget_type: unit
```

  **Ask the user for their Slack member ID** — do not guess it. They can find it
  in Slack: profile → ⋯ → Copy member ID.

- **Rebase**: if upstream main has moved, rebase the branch on the latest
  `upstream/main` and force-push. Resolve `models_unit_tests.yaml` conflicts by
  keeping both entries (yours and any newly added upstream ones).
