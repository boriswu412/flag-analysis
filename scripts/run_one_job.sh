#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source venv/bin/activate

protocol="$1"
config="$2"
t="${3:-1}"
job_id="${4:-?}"
total="${5:-?}"
name="$(basename "$config" .txt)"

if [[ -z "$protocol" || -z "$config" ]]; then
  echo "[$(date '+%H:%M:%S')] SKIP   ($job_id/$total) empty job line"
  exit 0
fi

if [[ ! -f "$protocol" ]]; then
  echo "[$(date '+%H:%M:%S')] SKIP   ($job_id/$total) missing protocol: $protocol"
  exit 0
fi

if [[ ! -f "$config" ]]; then
  echo "[$(date '+%H:%M:%S')] SKIP   ($job_id/$total) missing config: $config"
  exit 0
fi

echo "[$(date '+%H:%M:%S')] START  ($job_id/$total) t=$t $name"

set +e
python run_proof_protocol.py \
  --protocol "$protocol" \
  --config "$config" \
  --t "$t" \
  --metrics-dir results_txt \
  --quiet
rc=$?
set -e

echo "[$(date '+%H:%M:%S')] DONE   ($job_id/$total) t=$t $name"

exit "$rc"
