"""Shared test setup: point the harness at a real pack (Parakeet) without hardware.

Import this before any harness module. It puts the harness and the pack on
sys.path, selects the pack's models.json and loads its manifest.
"""
import os
from pathlib import Path
import sys

HARNESS = Path(__file__).resolve().parents[1]
PACK = HARNESS / "ports" / "parakeet"
for path in (PACK, HARNESS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("PORTFORGE_REPOSITORY_ROOT", str(HARNESS.parents[1]))

import models  # noqa: E402

models.use_registry(PACK / "models.json")

import port  # noqa: E402

PORT = port.load_port(PACK)


def configure_loop(loop):
    """Load the test pack into the controller module."""
    return loop.configure_port(PACK)


def write_harness(harness):
    """Populate a fake run harness the way the launcher freezes one."""
    import shutil
    harness = Path(harness)
    harness.mkdir(parents=True, exist_ok=True)
    for path in HARNESS.glob("*.py"):
        shutil.copyfile(path, harness / path.name)
    for folder in ("prompts", "guides"):
        shutil.copytree(HARNESS / folder, harness / folder, dirs_exist_ok=True)
    port.freeze(PORT, harness)
    return harness
