#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=parallel_limits.sh
source "$ROOT/scripts/parallel_limits.sh"
pl_init_dimacs_limits solve

PYTHON="$ROOT/venv/bin/python"
PROTOCOL="${1:?usage: run_export_and_solve.sh protocol.json config.txt [t]}"
CONFIG="${2:?usage: run_export_and_solve.sh protocol.json config.txt [t]}"
T="${3:-1}"

STEM="$(basename "$CONFIG" .txt)"
CNF_DIR="${CNF_DIR:-cnf_out/$STEM}"

echo "=== Phase 1: export DIMACS ==="
"$PYTHON" run_export_dimacs.py \
  --protocol "$PROTOCOL" \
  --config "$CONFIG" \
  --t "$T" \
  --cnf-dir "$CNF_DIR"

echo ""
echo "=== Phase 2: parallel solve ==="
bash "$ROOT/scripts/run_solve_paths.sh" "$CNF_DIR"
