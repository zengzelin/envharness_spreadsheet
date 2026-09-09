#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RAY_STATE_FILE="${RAY_STATE_FILE:-${ROOT}/runs/ray/ray_address.env}"
if [[ ! -f "${RAY_STATE_FILE}" ]]; then
  echo "Ray state file not found: ${RAY_STATE_FILE}" >&2
  echo "Run: bash rl/scripts/mpi_ray_up.sh" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${RAY_STATE_FILE}"
SPREADSHEETBENCH_DATA="${SPREADSHEETBENCH_DATA:-${ROOT}/experiments/spreadsheetbench/data/spreadsheetbench_verified_400}"
if [[ ! -f "${SPREADSHEETBENCH_DATA}/dataset.json" ]]; then
  echo "SpreadsheetBench dataset not found: ${SPREADSHEETBENCH_DATA}" >&2
  exit 1
fi

echo "[ray-smoke-submit] verifying ${RAY_ADDRESS}"
RAY_ADDRESS="${RAY_ADDRESS}" ray status >/dev/null

PYTHONPATH_VALUE="${ROOT}:${ROOT}/rl:${ROOT}/third_party/verl-agent${PYTHONPATH:+:${PYTHONPATH}}"
RUNTIME_ENV_JSON="$(
  ENVHARNESS_ROOT="${ROOT}" \
  SPREADSHEETBENCH_DATA="${SPREADSHEETBENCH_DATA}" \
  PYTHONPATH_VALUE="${PYTHONPATH_VALUE}" \
  python - <<'PY'
import json
import os

print(json.dumps({"env_vars": {
    "ENVHARNESS_ROOT": os.environ["ENVHARNESS_ROOT"],
    "SPREADSHEETBENCH_DATA": os.environ["SPREADSHEETBENCH_DATA"],
    "PYTHONPATH": os.environ["PYTHONPATH_VALUE"],
    "RAY_ADDRESS": "auto",
}}))
PY
)"

echo "[ray-smoke-submit] submitting through ${RAY_DASHBOARD_ADDRESS}"
ray job submit \
  --address="${RAY_DASHBOARD_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- bash -lc \
  'cd "$ENVHARNESS_ROOT" && python rl/scripts/smoke_spreadsheetbench_ray.py'
