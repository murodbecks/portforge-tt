"""Port packs: the model-specific half of a PortForge campaign.

A pack is a directory (see ports/template/) with:

  port.json     manifest validated here
  models.json   pinned checkpoint registry read by models.py
  evaluate.py   trusted evaluator: make_suite, reference, candidate, assess
  CONTRACT.md   interface and acceptance envelope the agents must meet
  TASK.md       ordered engineering plan for the worker
  BENCHMARK.md  optional corpus provenance and gate rationale
  SOURCES.md    optional model-specific public sources (papers, reference code, prior ports)
  corpus.json   optional frozen benchmark (otherwise pass --corpus)

Everything else (controller, agents, TT runner, reference jobs, guides) is shared.
When a campaign starts, the pack is frozen into the run's harness directory next
to the shared code, so a run always replays the exact evaluator it was graded by.
"""
import json
from pathlib import Path
import shutil

PACK_FILES = ("port.json", "models.json", "evaluate.py", "CONTRACT.md", "TASK.md")
OPTIONAL_DOCS = ("BENCHMARK.md", "SOURCES.md")
GUIDES = ("TT_GUIDE.md", "LESSONS.md")
TT_PRECISIONS = ("bf16", "fp32", "fp16", "bfp8_b")

DEFAULT_OBJECTIVE = (
    "Deliver ONE production-level reusable {title} TTNN source packaged in tt-metal style (tt/, demo/, "
    "reference/, tests/, benchmarks/, docs/ with SPDX headers) suitable for an upstream tt-metal PR. Establish "
    "measured baselines FIRST and preserve them (PyTorch on the TT host; CUDA reference timings are the NVIDIA "
    "comparison context); pass smoke, then bringup against the FP32 oracle; profile and optimize against the "
    "recorded baseline with matched same-precision comparisons; assemble the tt-metal-style package and "
    "portable tests; then freeze for full validation. A first bringup pass alone is not the goal, and "
    "optimization claims without a recorded baseline are invalid. Advance independent implementation when "
    "runtime diagnosis blocks final evaluation. Follow LESSONS.md.")
DEFAULT_PRECISION_POLICY = (
    "BF16 primary; investigate supported BFP8_B with honest quality, latency and memory tradeoffs. Missing "
    "cells are not support or passes; BFP8_B is block float, not IEEE FP8. Do not inherit CUDA precision results.")


def _strings(value, field):
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"port.json {field} must be a nonempty list of strings")
    return list(value)


def load_port(directory):
    """Validate a pack (or a frozen harness holding one) and return its resolved manifest."""
    root = Path(directory).resolve()
    missing = [name for name in PACK_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"{root} is not a port pack; missing {', '.join(missing)}")
    manifest = json.loads((root / "port.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported port.json schema")
    registry = json.loads((root / "models.json").read_text())["models"]
    key = manifest.get("key")
    delivery = _strings(manifest.get("delivery_models", [key]), "delivery_models")
    if key not in registry or key not in delivery or any(name not in registry for name in delivery):
        raise ValueError("port.json key must be listed in delivery_models, and every delivery model must be a models.json key")
    precisions = _strings(manifest.get("precisions", list(TT_PRECISIONS)), "precisions")
    delivery_precisions = _strings(manifest.get("delivery_precisions", ["bf16"]), "delivery_precisions")
    if any(p not in TT_PRECISIONS for p in precisions) or any(p not in precisions for p in delivery_precisions):
        raise ValueError("precisions must be TT formats and include every delivery precision")
    corpus = manifest.get("corpus") or {}
    digest = corpus.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("port.json corpus.sha256 must pin the frozen benchmark")
    corpus_file = corpus.get("file")
    if corpus_file is not None and (not isinstance(corpus_file, str) or Path(corpus_file).name != corpus_file):
        raise ValueError("corpus.file must be a file name inside the pack")
    title = manifest.get("title") or key
    reference = manifest.get("reference") or {}
    return {
        "dir": root, "key": key, "title": title, "task": manifest.get("task", ""),
        "repo_id": registry[key]["repo_id"], "delivery_models": delivery,
        "precisions": precisions, "delivery_precisions": delivery_precisions,
        "corpus_sha256": digest, "corpus_path": root / corpus_file if corpus_file else None,
        "objective": manifest.get("objective") or DEFAULT_OBJECTIVE.format(title=title),
        "precision_policy": manifest.get("precision_policy") or DEFAULT_PRECISION_POLICY,
        "docs": [name for name in PACK_FILES + OPTIONAL_DOCS if name.endswith(".md") and (root / name).is_file()],
        "reference_packages": list(reference.get("packages", [])),
        "reference_imports": list(reference.get("imports", [])),
    }


def seed_documents(port, guides_dir):
    """Agent-facing instruction files for a fresh workspace: pack docs plus shared guides."""
    documents = {name: (port["dir"] / name).read_text() for name in port["docs"]}
    for name in GUIDES:
        documents[name] = (Path(guides_dir) / name).read_text()
    return documents


def freeze(port, harness):
    """Copy the pack into a run's harness directory, beside the shared code."""
    harness = Path(harness)
    for name in PACK_FILES + OPTIONAL_DOCS:
        if (port["dir"] / name).is_file():
            shutil.copyfile(port["dir"] / name, harness / name)
