#!/usr/bin/env bash
# Rebuild everything from the seeds (about three minutes).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data results/tables results/figures
python src/simulate.py data
python src/forecast.py
python src/allocate.py
python src/reorder_sim.py
python src/figures.py
python -m pytest -q tests
