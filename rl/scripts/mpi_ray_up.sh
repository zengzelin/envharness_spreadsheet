#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RAY_RUN_DIR="${RAY_RUN_DIR:-${ROOT}/runs/ray}"
mkdir -p "${RAY_RUN_DIR}"

LOG_FILE="${LOG_FILE:-${RAY_RUN_DIR}/ray_up_${RUN_TS}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[spreadsheet-ray-up] starting script=${BASH_SOURCE[0]} log=${LOG_FILE}"

SOURCE_HOSTFILE="${SOURCE_HOSTFILE:-/etc/mpi/hostfile}"
HOSTFILE="${HOSTFILE:-${RAY_RUN_DIR}/hostfile_spreadsheetbench}"
INCLUDE_LAUNCHER="${INCLUDE_LAUNCHER:-1}"
REQUESTED_NNODES="${NNODES:-}"
if [[ -n "${REQUESTED_NNODES}" && ! "${REQUESTED_NNODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NNODES must be a positive integer, got ${REQUESTED_NNODES}" >&2
  exit 2
fi

if [[ "${REQUESTED_NNODES}" == "1" ]]; then
  LOCAL_HOST="${MASTER_ADDR:-${__POD_IP__:-${__HOST_IP__:-}}}"
  if [[ -z "${LOCAL_HOST}" ]]; then
    LOCAL_HOST="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  fi
  if [[ -z "${LOCAL_HOST}" ]]; then
    LOCAL_HOST="$(hostname)"
  fi
  printf '%s slots=1\n' "${LOCAL_HOST}" > "${HOSTFILE}"
elif [[ "${INCLUDE_LAUNCHER}" == "1" ]]; then
  awk '!seen[$1]++ {print}' "${SOURCE_HOSTFILE}" > "${HOSTFILE}"
else
  awk '$1 !~ /launcher/ && !seen[$1]++ {print}' "${SOURCE_HOSTFILE}" > "${HOSTFILE}"
fi
if [[ ! -s "${HOSTFILE}" ]]; then
  echo "No hosts found in ${SOURCE_HOSTFILE}" >&2
  exit 1
fi
sed -i 's/slots=[0-9][0-9]*/slots=1/g' "${HOSTFILE}"
if [[ -n "${REQUESTED_NNODES}" ]]; then
  SELECTED_HOSTFILE="${HOSTFILE}.selected"
  head -n "${REQUESTED_NNODES}" "${HOSTFILE}" > "${SELECTED_HOSTFILE}"
  mv "${SELECTED_HOSTFILE}" "${HOSTFILE}"
fi

NNODES="$(wc -l < "${HOSTFILE}" | awk '{print $1}')"
if [[ -n "${REQUESTED_NNODES}" && "${NNODES}" -ne "${REQUESTED_NNODES}" ]]; then
  echo "Requested NNODES=${REQUESTED_NNODES}, but only ${NNODES} host(s) are available" >&2
  exit 1
fi
HEAD_HOST="$(awk 'NR == 1 {print $1}' "${HOSTFILE}")"
HEAD_HOST_IP="$(getent hosts "${HEAD_HOST}" 2>/dev/null | awk '{print $1; exit}' || true)"
LAUNCHER_IP="${__POD_IP__:-${__HOST_IP__:-}}"
if [[ -z "${LAUNCHER_IP}" ]]; then
  LAUNCHER_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
fi
if [[ -z "${LAUNCHER_IP}" ]]; then
  LAUNCHER_IP="${MASTER_ADDR:-}"
fi
if [[ -z "${LAUNCHER_IP}" ]]; then
  echo "Unable to determine launcher IP; set __POD_IP__, __HOST_IP__, or MASTER_ADDR" >&2
  exit 1
fi
if [[ -z "${HEAD_HOST_IP}" ]]; then
  HEAD_HOST_IP="${LAUNCHER_IP}"
fi

if [[ "${INCLUDE_LAUNCHER}" == "1" ]] && grep -q launcher "${HOSTFILE}"; then
  MASTER_ADDR="${MASTER_ADDR:-${LAUNCHER_IP}}"
else
  MASTER_ADDR="${MASTER_ADDR:-${HEAD_HOST_IP}}"
fi
GPUS_PER_NODE="${GPUS_PER_NODE:-${N_GPUS_PER_NODE:-8}}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray}"
MPI_TCP_IFACE="${MPI_TCP_IFACE:-bond1}"
export GPUS_PER_NODE RAY_DASHBOARD_PORT RAY_TEMP_DIR

echo "[spreadsheet-ray-up] log=${LOG_FILE}"
echo "[spreadsheet-ray-up] nnodes=${NNODES} head=${HEAD_HOST} master=${MASTER_ADDR} gpus_per_node=${GPUS_PER_NODE}"
if [[ "${NNODES}" -eq 1 ]]; then
  echo "[spreadsheet-ray-up] launch_mode=single-node"
else
  echo "[spreadsheet-ray-up] launch_mode=multi-node"
fi
cat "${HOSTFILE}"

if [[ "${RAY_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  echo "[spreadsheet-ray-up] dry run complete"
  exit 0
fi

if [[ "${NNODES}" -eq 1 ]]; then
  OMPI_COMM_WORLD_RANK=0 bash "${SCRIPT_DIR}/mpi_ray_node.sh" "${MASTER_ADDR}"
else
  MPI_ARGS=(
    -v --allow-run-as-root
    --hostfile "${HOSTFILE}"
    --bind-to none
    --map-by slot
    --mca routed direct
    --mca btl_tcp_if_include "${MPI_TCP_IFACE}"
    --mca oob_tcp_if_include "${MPI_TCP_IFACE}"
    -x PATH -x LD_LIBRARY_PATH -x GPUS_PER_NODE
    -x RAY_DASHBOARD_PORT -x RAY_TEMP_DIR
  )
  if [[ -n "${RAY_SYSTEM_CONFIG:-}" ]]; then
    export RAY_SYSTEM_CONFIG
    MPI_ARGS+=(-x RAY_SYSTEM_CONFIG)
  fi
  mpirun "${MPI_ARGS[@]}" bash "${SCRIPT_DIR}/mpi_ray_node.sh" "${MASTER_ADDR}"
fi

RAY_ADDRESS="${MASTER_ADDR}:6379"
READY=0
for _attempt in $(seq 1 60); do
  if RAY_ADDRESS="${RAY_ADDRESS}" ray status >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "${READY}" != "1" ]]; then
  echo "Ray cluster failed readiness check at ${RAY_ADDRESS}" >&2
  exit 1
fi

RAY_STATE_FILE="${RAY_STATE_FILE:-${RAY_RUN_DIR}/ray_address.env}"
{
  printf 'export MASTER_ADDR=%q\n' "${MASTER_ADDR}"
  printf 'export RAY_ADDRESS=%q\n' "${RAY_ADDRESS}"
  printf 'export RAY_DASHBOARD_ADDRESS=%q\n' "http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}"
  printf 'export NNODES=%q\n' "${NNODES}"
  printf 'export GPUS_PER_NODE=%q\n' "${GPUS_PER_NODE}"
} > "${RAY_STATE_FILE}"

RAY_ADDRESS="${RAY_ADDRESS}" ray status
echo "[spreadsheet-ray-up] Ray cluster is ready"
echo "[spreadsheet-ray-up] state_file=${RAY_STATE_FILE}"
echo "[spreadsheet-ray-up] next: bash rl/scripts/submit_spreadsheetbench_grpo.sh"
