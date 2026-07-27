#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate zi2zi-jit
python webui.py "$@"
