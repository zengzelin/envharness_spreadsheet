#!/usr/bin/env bash
set -euo pipefail

MASTER_ADDR="$1"
RANK="${OMPI_COMM_WORLD_RANK:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
LOCAL_IP="${__HOST_IP__:-$(hostname -I | awk '{print $1}')}"
if [[ "${RANK}" -eq 0 ]]; then
  LOCAL_IP="${MASTER_ADDR}"
fi

export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${no_proxy:-}"
export RAY_USAGE_STATS_ENABLED=0
export RAY_RUNTIME_ENV_WORKING_DIR_CACHE_SIZE_GB="${RAY_RUNTIME_ENV_WORKING_DIR_CACHE_SIZE_GB:-20}"
export RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB="${RAY_RUNTIME_ENV_PIP_CACHE_SIZE_GB:-10}"
export RAY_RUNTIME_ENV_AGENT_TIMEOUT_MS="${RAY_RUNTIME_ENV_AGENT_TIMEOUT_MS:-120000}"

RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_HEAD_EXTRA_ARGS=()
if [[ -n "${RAY_SYSTEM_CONFIG:-}" ]]; then
  RAY_HEAD_EXTRA_ARGS+=(--system-config="${RAY_SYSTEM_CONFIG}")
fi

ray stop --force >/dev/null 2>&1 || true
sleep 2

echo "[spreadsheet-ray-node] rank=${RANK} master=${MASTER_ADDR} local_ip=${LOCAL_IP}"

if [[ "${RANK}" -eq 0 ]]; then
  ray start --head \
    --port=6379 \
    --node-ip-address="${LOCAL_IP}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --temp-dir="${RAY_TEMP_DIR}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    "${RAY_HEAD_EXTRA_ARGS[@]}"
else
  for _attempt in $(seq 1 60); do
    if timeout 1 bash -c "</dev/tcp/${MASTER_ADDR}/6379" 2>/dev/null; then
      break
    fi
    if [[ "${_attempt}" -eq 60 ]]; then
      echo "Ray head ${MASTER_ADDR}:6379 did not become reachable" >&2
      exit 1
    fi
    sleep 2
  done
  ray start \
    --address="${MASTER_ADDR}:6379" \
    --num-gpus="${GPUS_PER_NODE}" \
    --node-ip-address="${LOCAL_IP}"
fi

echo "[spreadsheet-ray-node] rank=${RANK} started"
