#!/usr/bin/env bash
# Puts the model + fossil data into ./data if it isn't already there.
# Data either lives in the repo (committed under data/) or is downloaded from
# DATA_URL (set in .devcontainer/devcontainer.json -> containerEnv), e.g. a
# GitHub Release asset: https://github.com/<org>/<repo>/releases/download/v1/student_data.zip
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f data/model/model_params_config.json ]; then
    echo "Data already present in ./data — nothing to download."
    exit 0
fi

if [ -z "${DATA_URL:-}" ]; then
    echo "WARNING: ./data is empty and DATA_URL is not set."
    echo "         Either commit the data bundle to data/ or set DATA_URL in .devcontainer/devcontainer.json."
    exit 0
fi

echo "Downloading data bundle from $DATA_URL"
curl -fL --retry 3 "$DATA_URL" -o /tmp/student_data.zip
mkdir -p data
unzip -oq /tmp/student_data.zip -d data
rm -f /tmp/student_data.zip
echo "Data unpacked into ./data"
