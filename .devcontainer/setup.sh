#!/usr/bin/env bash
# Runs once when the codespace (or its prebuild) is created.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [1/5] System libraries needed by VTK / PyVista (no display required)"
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends \
    libgl1 libxrender1 libglib2.0-0 libgomp1 unzip curl

echo "==> [2/5] Python packages (CPU-only PyTorch)"
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt

echo "==> [3/5] NSM toolkit (installed WITHOUT its own dependencies so it can't pull in CUDA torch)"
# TODO: pin to a specific commit once the demo is stable, e.g.
#   git+https://github.com/3D-fossils-Haag/nsm.git@<commit-sha>
python -m pip install --no-deps "${NSM_PIP_SPEC:-git+https://github.com/3D-fossils-Haag/nsm.git}"

echo "==> [4/5] Jupyter kernel"
python -m ipykernel install --user --name fossil-explorer --display-name "Fossil Explorer"

echo "==> [5/5] Fossil data"
bash scripts/get_data.sh

echo "==> Self-check"
python -c "import helpers; helpers.self_check()" || echo "WARNING: self-check failed — see messages above."
echo "==> Setup complete."
