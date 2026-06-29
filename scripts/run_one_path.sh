#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR  venv not found. Run: bash scripts/setup_server.sh" >&2
  exit 1
fi

cnf_dir="$1"
path_tag="$2"
job_id="${3:-?}"
total="${4:-?}"

if [[ -z "$cnf_dir" || -z "$path_tag" ]]; then
  exit 0
fi

if [[ ! -f "$cnf_dir/$path_tag.cnf" ]]; then
  echo "SKIP   missing $cnf_dir/$path_tag.cnf" >&2
  exit 0
fi

set +e
"$PYTHON" run_solve_dimacs.py \
  --cnf-dir "$cnf_dir" \
  --path-tag "$path_tag" \
  --parse-only \
  --job-id "$job_id" \
  --total "$total"
rc=$?
set -e

exit "$rc"
