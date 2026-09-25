#!/usr/bin/env bash
# Build the FP32 reference environment for one port pack on the reference host.
#
#   bash setup_reference.sh <port.json> [reference_root]
#
# Copy the pack's port.json to the reference host first. No sudo, driver changes
# or global installs. CUDA is preferred for the FP32 oracle and required for the
# BF16 reference study; a CPU-only host is accepted and recorded.
set -euo pipefail
port_json="${1:?usage: setup_reference.sh <port.json> [reference_root]}"
reference_root="${2:-$HOME/portforge}"
torch_index="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"  # use .../whl/cpu on CPU-only hosts
mkdir -p "$reference_root"
python3 -m venv "$reference_root/.venv"
reference_python="$reference_root/.venv/bin/python"
"$reference_python" -m pip install --only-binary=:all: --index-url https://pypi.org/simple pip==26.2.1 setuptools==84.0.0
"$reference_python" -m pip install --index-url "$torch_index" torch==2.14.0 --report "$reference_root/torch-install.json"
packages=()
while IFS= read -r package; do
  [ -n "$package" ] && packages+=("$package")
done < <("$reference_python" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1])).get("reference", {}).get("packages", [])))' "$port_json")
if [ "${#packages[@]}" -gt 0 ]; then
  "$reference_python" -m pip install --index-url https://pypi.org/simple "${packages[@]}" --report "$reference_root/reference-install.json"
fi
"$reference_python" -m pip check
"$reference_python" -m pip freeze > "$reference_root/requirements.lock.txt"
"$reference_python" - "$port_json" <<'PY'
import importlib, json, sys
import torch
for name in json.load(open(sys.argv[1])).get("reference", {}).get("imports", []):
    importlib.import_module(name)  # the pack's reference implementation must import
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(32, 32, device=device)
assert torch.isfinite(x @ x).all()
if device == "cuda":
    print({"torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0), "bf16": torch.cuda.is_bf16_supported()})
else:
    print({"torch": torch.__version__, "device": "cpu (recorded; CUDA preferred)"})
PY
