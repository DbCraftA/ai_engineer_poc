#!/usr/bin/env bash
set -euo pipefail

python -m pytest -m "not gpu and not slow"

