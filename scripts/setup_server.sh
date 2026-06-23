#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3-venv python3-pip cryptominisat
else
    echo "apt-get not found; install python3-venv, python3-pip, and cryptominisat manually."
fi

python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install -r requirements.txt

if command -v cryptominisat5 >/dev/null 2>&1; then
    cryptominisat5 --version
else
    echo "Warning: cryptominisat5 not found on PATH."
    exit 1
fi

echo "Setup complete. Activate with: source venv/bin/activate"
