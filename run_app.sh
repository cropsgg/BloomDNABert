#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  echo "Creating virtual environment..."
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  else
    python3 -m venv .venv
  fi
  PIP_DISABLE_PIP_VERSION_CHECK=1 .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/python -u app.py
