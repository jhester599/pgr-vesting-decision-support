#!/usr/bin/env bash
set -euo pipefail

START=$(python -c "import time; print(time.time())")
python -m pytest --tb=no -q "$@" > /dev/null 2>&1
python -c "import time; start=float(${START}); print(f'elapsed_seconds={time.time()-start:.1f}')"
