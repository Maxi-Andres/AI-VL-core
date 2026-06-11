#!/usr/bin/env bash
# setup.sh — create the project virtual environment and install dependencies.
#
# Goal: `git clone <repo>` + `./setup.sh` = a working PoC (VLM + YOLO), WITHOUT
# touching the system Python. The venv itself is NOT committed (it is
# machine-specific and git-ignored); this script rebuilds it from
# requirements.txt on any machine.
#
# Why a venv at all? Many systems (Debian/Ubuntu, and this one with Python 3.14)
# ship an "externally-managed" Python (PEP 668) where `pip install` is blocked.
# A venv sidesteps that cleanly and reproducibly.
#
# Usage:
#   ./setup.sh                 # uses `python3`
#   PYTHON=python3.12 ./setup.sh   # pick a specific interpreter
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "[setup] ERROR: '$PYTHON' not found. Install Python 3 (or set PYTHON=...)." >&2
  exit 1
fi
echo "[setup] Python: $("$PYTHON" --version 2>&1)"

# 1) Create the venv. Try the normal way first; if this platform's venv lacks
#    ensurepip (common on 'externally-managed' Pythons), fall back to a pip-less
#    venv that we bootstrap with get-pip.py — no apt/sudo required.
need_bootstrap=0
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV" 2>/dev/null || need_bootstrap=1
fi
if [ "$need_bootstrap" = 1 ] || ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "[setup] No pip in the venv (no ensurepip) — bootstrapping with get-pip.py."
  rm -rf "$VENV"
  "$PYTHON" -m venv --without-pip "$VENV"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python"
  else
    echo "[setup] ERROR: need curl or wget to bootstrap pip." >&2
    exit 1
  fi
fi

# 2) Install dependencies into the venv (requests = VLM path, ultralytics = YOLO).
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements.txt

echo
echo "[setup] Done."
echo "[setup] Activate it:        source $VENV/bin/activate   (then: python menu.py)"
echo "[setup] Or run directly:    $VENV/bin/python menu.py"
