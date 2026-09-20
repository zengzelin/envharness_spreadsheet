#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${PY:-$(command -v python || command -v python3)}"
RAY_STATE_FILE="${RAY_STATE_FILE:-${ROOT}/runs/ray/ray_address.env}"
if [[ -f "${RAY_STATE_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${RAY_STATE_FILE}"
elif [[ -n "${RAY_ADDRESS:-}" && -n "${RAY_DASHBOARD_ADDRESS:-${RAY_JOB_ADDRESS:-}}" ]]; then
  RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-${RAY_JOB_ADDRESS}}"
  MASTER_ADDR="${MASTER_ADDR:-${RAY_ADDRESS%%:*}}"
  NNODES="${NNODES:-1}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-${N_GPUS_PER_NODE:-8}}"
  echo "[spreadsheet-train-submit] using explicit Ray addresses"
else
  echo "Ray state file not found: ${RAY_STATE_FILE}" >&2
  echo "Run: bash rl/scripts/mpi_ray_up.sh, or set RAY_ADDRESS and RAY_DASHBOARD_ADDRESS" >&2
  exit 1
fi

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${N_GPUS_PER_NODE}}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-${RAY_JOB_ADDRESS:-}}"
if [[ -z "${RAY_DASHBOARD_ADDRESS}" ]]; then
  echo "RAY_DASHBOARD_ADDRESS is required" >&2
  exit 1
fi
SPREADSHEETBENCH_DATA_FORMAT="${SPREADSHEETBENCH_DATA_FORMAT:-spreadsheetbench}"
case "${SPREADSHEETBENCH_DATA_FORMAT}" in
  spreadsheetbench|spreadsheet_bench)
    SPREADSHEETBENCH_DATA="${SPREADSHEETBENCH_DATA:-${ROOT}/experiments/spreadsheetbench/data/spreadsheetbench_verified_400}"
    if [[ ! -f "${SPREADSHEETBENCH_DATA}/dataset.json" ]]; then
      echo "SpreadsheetBench dataset not found: ${SPREADSHEETBENCH_DATA}" >&2
      exit 1
    fi
    ;;
  spreadsheet_rl|spreadsheet-rl|spreadsheetrl)
    SPREADSHEET_RL_DATA_ROOT="${SPREADSHEET_RL_DATA_ROOT:-${SPREADSHEETBENCH_DATA:-${ROOT}/experiments/spreadsheetbench/data/Spreadsheet-RL}}"
    SPREADSHEET_RL_TRAIN_FILE="${SPREADSHEET_RL_TRAIN_FILE:-train_hermes.parquet}"
    SPREADSHEET_RL_VAL_FILE="${SPREADSHEET_RL_VAL_FILE:-test_verified_hermes.parquet}"
    SPREADSHEETBENCH_DATA="${SPREADSHEET_RL_DATA_ROOT}"
    SPREADSHEET_RL_TRAIN_PATH="${SPREADSHEET_RL_TRAIN_FILE}"
    SPREADSHEET_RL_VAL_PATH="${SPREADSHEET_RL_VAL_FILE}"
    [[ "${SPREADSHEET_RL_TRAIN_PATH}" == /* ]] || SPREADSHEET_RL_TRAIN_PATH="${SPREADSHEET_RL_DATA_ROOT}/${SPREADSHEET_RL_TRAIN_FILE}"
    [[ "${SPREADSHEET_RL_VAL_PATH}" == /* ]] || SPREADSHEET_RL_VAL_PATH="${SPREADSHEET_RL_DATA_ROOT}/${SPREADSHEET_RL_VAL_FILE}"
    if [[ ! -f "${SPREADSHEET_RL_TRAIN_PATH}" ]]; then
      echo "Spreadsheet-RL train split not found: ${SPREADSHEET_RL_TRAIN_PATH}" >&2
      exit 1
    fi
    if [[ ! -f "${SPREADSHEET_RL_VAL_PATH}" ]]; then
      echo "Spreadsheet-RL val split not found: ${SPREADSHEET_RL_VAL_PATH}" >&2
      exit 1
    fi
    ;;
  *)
    echo "SPREADSHEETBENCH_DATA_FORMAT must be spreadsheetbench or spreadsheet_rl, got ${SPREADSHEETBENCH_DATA_FORMAT}" >&2
    exit 2
    ;;
esac
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  command -v soffice >/dev/null 2>&1 || { echo "soffice is required" >&2; exit 1; }
fi

MODE="${MODE:-smoke}"
if [[ "${MODE}" == "full" || "${MODE}" == "diagnostic" ]]; then
  DEFAULT_MODEL="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/luban/common/models/Qwen3-4B-Thinking-2507"
else
  DEFAULT_MODEL="${ROOT}/../llm_model/Qwen2.5-1.5B-Instruct"
  if [[ ! -f "${DEFAULT_MODEL}/config.json" ]]; then
    DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
  fi
fi
MODEL="${MODEL:-${DEFAULT_MODEL}}"
export WANDB_MODE="${WANDB_MODE:-online}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console','wandb','tensorboard']}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/grpo_spreadsheetbench_${MODE}_${RUN_TS}}"
EXP_NAME="${EXP_NAME:-grpo_spreadsheetbench_${MODE}_${RUN_TS}}"
WANDB_PROJECT="${WANDB_PROJECT:-envharness_rl_spreadsheetbench}"
WANDB_NAME="${WANDB_NAME:-${EXP_NAME}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}}"
WANDB_DIR="${WANDB_DIR:-${LOG_DIR}/wandb}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${LOG_DIR}/tensorboard}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${RUN_DIR}/rollouts/verl}"
SPREADSHEETBENCH_TRAJECTORY_DIR="${SPREADSHEETBENCH_TRAJECTORY_DIR:-${RUN_DIR}/rollouts/env}"
if [[ "${MODE}" == "diagnostic" ]]; then
  SPREADSHEETBENCH_HISTORY_MODE="${SPREADSHEETBENCH_HISTORY_MODE:-compact}"
else
  SPREADSHEETBENCH_HISTORY_MODE="${SPREADSHEETBENCH_HISTORY_MODE:-full}"
fi
SPREADSHEETBENCH_HISTORY_ACTION_CHARS="${SPREADSHEETBENCH_HISTORY_ACTION_CHARS:-1200}"
SPREADSHEETBENCH_HISTORY_OBS_CHARS="${SPREADSHEETBENCH_HISTORY_OBS_CHARS:-2000}"
SPREADSHEETBENCH_PYTHON_ERROR_PENALTY="${SPREADSHEETBENCH_PYTHON_ERROR_PENALTY:--0.05}"
SPREADSHEETBENCH_SYNTAX_ERROR_PENALTY="${SPREADSHEETBENCH_SYNTAX_ERROR_PENALTY:--0.1}"
SPREADSHEETBENCH_TOOL_SET="${SPREADSHEETBENCH_TOOL_SET:-python}"
SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS="${SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS:-600}"
SPREADSHEETBENCH_PHASE_HEARTBEAT_SECONDS="${SPREADSHEETBENCH_PHASE_HEARTBEAT_SECONDS:-60}"
SPREADSHEETBENCH_TOOL_SET="${SPREADSHEETBENCH_TOOL_SET//-/_}"
case "${SPREADSHEETBENCH_TOOL_SET}" in
  python|native_read|native_basic) ;;
  *)
    echo "SPREADSHEETBENCH_TOOL_SET must be python, native_read, or native_basic, got ${SPREADSHEETBENCH_TOOL_SET}" >&2
    exit 2
    ;;
esac
for timeout_name in \
  SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS \
  SPREADSHEETBENCH_PHASE_HEARTBEAT_SECONDS; do
  timeout_value="${!timeout_name}"
  if [[ ! "${timeout_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${timeout_name} must be a positive integer, got ${timeout_value}" >&2
    exit 2
  fi
done
mkdir -p "${LOG_DIR}" "${WANDB_DIR}" "${TENSORBOARD_DIR}" \
  "${ROLLOUT_DATA_DIR}" "${SPREADSHEETBENCH_TRAJECTORY_DIR}"
JOB_LOG="${JOB_LOG:-${LOG_DIR}/submit.log}"
exec > >(tee -a "${JOB_LOG}") 2>&1

export no_proxy="127.0.0.1,localhost,${MASTER_ADDR:-},${no_proxy:-}"
export WANDB_MODE TRAINER_LOGGER PYTHONUNBUFFERED
export PYTHONFAULTHANDLER VLLM_USE_V1 RUN_TS
export MODE RUN_ROOT RUN_DIR EXP_NAME WANDB_PROJECT WANDB_NAME LOG_DIR
export WANDB_DIR TENSORBOARD_DIR ROLLOUT_DATA_DIR SPREADSHEETBENCH_TRAJECTORY_DIR
export SPREADSHEETBENCH_HISTORY_MODE
export SPREADSHEETBENCH_HISTORY_ACTION_CHARS
export SPREADSHEETBENCH_HISTORY_OBS_CHARS
export SPREADSHEETBENCH_PYTHON_ERROR_PENALTY
export SPREADSHEETBENCH_SYNTAX_ERROR_PENALTY
export SPREADSHEETBENCH_TOOL_SET
export SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS
export SPREADSHEETBENCH_PHASE_HEARTBEAT_SECONDS
export SPREADSHEETBENCH_DATA_FORMAT SPREADSHEET_RL_DATA_ROOT
export SPREADSHEET_RL_TRAIN_FILE SPREADSHEET_RL_VAL_FILE

if [[ "${DRY_RUN:-0}" != "1" ]]; then
"${PY}" - <<'PY'
import importlib
import os

for name in ("datasets", "openpyxl", "ray", "torch", "transformers", "vllm"):
    importlib.import_module(name)
if "wandb" in os.environ.get("TRAINER_LOGGER", ""):
    importlib.import_module("wandb")
if "tensorboard" in os.environ.get("TRAINER_LOGGER", ""):
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
print("[spreadsheet-train-submit] Python dependencies OK")
PY
fi

echo "[spreadsheet-train-submit] verifying ${RAY_ADDRESS}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  RAY_ADDRESS="${RAY_ADDRESS}" ray status >/dev/null
fi

PYTHONPATH_VALUE="${ROOT}:${ROOT}/rl:${ROOT}/third_party/verl-agent${PYTHONPATH:+:${PYTHONPATH}}"
RUNTIME_ENV_JSON="$(
  ENVHARNESS_ROOT="${ROOT}" \
  SPREADSHEETBENCH_DATA="${SPREADSHEETBENCH_DATA}" \
  SPREADSHEETBENCH_DATA_FORMAT="${SPREADSHEETBENCH_DATA_FORMAT}" \
  SPREADSHEET_RL_DATA_ROOT="${SPREADSHEET_RL_DATA_ROOT:-}" \
  SPREADSHEET_RL_TRAIN_FILE="${SPREADSHEET_RL_TRAIN_FILE:-}" \
  SPREADSHEET_RL_VAL_FILE="${SPREADSHEET_RL_VAL_FILE:-}" \
  PYTHONPATH_VALUE="${PYTHONPATH_VALUE}" \
  NNODES="${NNODES}" \
  GPUS_PER_NODE="${GPUS_PER_NODE}" \
  N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
  MODE="${MODE}" \
  MODEL="${MODEL}" \
  WANDB_MODE="${WANDB_MODE}" \
  TRAINER_LOGGER="${TRAINER_LOGGER}" \
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
  PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER}" \
  VLLM_USE_V1="${VLLM_USE_V1}" \
  RUN_TS="${RUN_TS}" \
  "${PY}" - <<'PY'
import json
import os

forward = [
    "ENVHARNESS_ROOT", "SPREADSHEETBENCH_DATA", "SPREADSHEETBENCH_DATA_FORMAT",
    "SPREADSHEET_RL_DATA_ROOT", "SPREADSHEET_RL_TRAIN_FILE",
    "SPREADSHEET_RL_VAL_FILE", "NNODES", "GPUS_PER_NODE",
    "N_GPUS_PER_NODE",
    "MODE", "MODEL", "TRAIN_BS", "VAL_BS", "GROUP_N", "PPO_MINI_BS",
    "EPOCHS", "TOTAL_TRAINING_STEPS", "N_GPUS_PER_NODE", "TP", "MAX_STEPS", "HISTORY_LENGTH",
    "TEST_FREQ", "VAL_BEFORE", "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH",
    "APPLY_CHAT_TEMPLATE_ENABLE_THINKING",
    "NUM_CPUS_PER_ENV", "PPO_MICRO_BS_PER_GPU", "LOG_PROB_MICRO_BS_PER_GPU",
    "GPU_MEM_UTIL", "ROLLOUT_TOP_K", "ROLLOUT_TOP_P", "ROLLOUT_TEMPERATURE",
    "ROLLOUT_ENABLE_CHUNKED_PREFILL", "ROLLOUT_MAX_MODEL_LEN",
    "ROLLOUT_MAX_NUM_BATCHED_TOKENS", "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU",
    "LOG_PROB_MAX_TOKEN_LEN_PER_GPU",
    "VAL_TEMPERATURE", "VAL_DO_SAMPLE", "ACTOR_LR", "KL_LOSS_COEF", "ENTROPY_COEFF",
    "USE_INVALID_ACTION_PENALTY", "INVALID_ACTION_PENALTY_COEF",
    "SAVE_FREQ", "EXP_NAME", "WANDB_PROJECT", "RESUME_FROM",
    "EXTRA_HYDRA", "VLLM_ATTENTION_BACKEND", "ENVHARNESS_DISABLE_THINKING",
    "SPREADSHEETBENCH_HISTORY_MODE", "SPREADSHEETBENCH_HISTORY_ACTION_CHARS",
    "SPREADSHEETBENCH_HISTORY_OBS_CHARS",
    "SPREADSHEETBENCH_PYTHON_ERROR_PENALTY",
    "SPREADSHEETBENCH_SYNTAX_ERROR_PENALTY",
    "SPREADSHEETBENCH_TOOL_SET",
    "SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS",
    "SPREADSHEETBENCH_PHASE_HEARTBEAT_SECONDS",
    "WANDB_BASE_URL", "WANDB_API_KEY", "WANDB_MODE", "WANDB_DIR", "WANDB_NAME",
    "TENSORBOARD_DIR", "TRAINER_LOGGER", "LOG_VAL_GENERATIONS", "LOG_DIR",
    "ROLLOUT_DATA_DIR", "SPREADSHEETBENCH_TRAJECTORY_DIR", "RUN_ROOT", "RUN_DIR",
    "RUN_TS", "PYTHONUNBUFFERED", "PYTHONFAULTHANDLER", "VLLM_USE_V1", "no_proxy",
]
env_vars = {key: os.environ[key] for key in forward if key in os.environ}
env_vars["PYTHONPATH"] = os.environ["PYTHONPATH_VALUE"]
env_vars["RAY_ADDRESS"] = "auto"
print(json.dumps({
    "excludes": [
        ".git",
        "runs",
        "wandb",
        "tensorboard_log",
        ".pytest_cache",
        "**/__pycache__",
        "third_party/verl-agent/.git",
    ],
    "env_vars": env_vars,
}))
PY
)"

echo "[spreadsheet-train-submit] mode=${MODE:-smoke} model=${MODEL}"
echo "[spreadsheet-train-submit] data_format=${SPREADSHEETBENCH_DATA_FORMAT} data=${SPREADSHEETBENCH_DATA}"
if [[ "${SPREADSHEETBENCH_DATA_FORMAT}" == "spreadsheet_rl" || "${SPREADSHEETBENCH_DATA_FORMAT}" == "spreadsheet-rl" || "${SPREADSHEETBENCH_DATA_FORMAT}" == "spreadsheetrl" ]]; then
  echo "[spreadsheet-train-submit] train_split=${SPREADSHEET_RL_TRAIN_FILE} val_split=${SPREADSHEET_RL_VAL_FILE}"
fi
echo "[spreadsheet-train-submit] job_log=${JOB_LOG}"
echo "[spreadsheet-train-submit] working_dir=${ROOT}"
echo "[spreadsheet-train-submit] nnodes=${NNODES} gpus_per_node=${GPUS_PER_NODE} n_gpus_per_node=${N_GPUS_PER_NODE}"
echo "[spreadsheet-train-submit] history_mode=${SPREADSHEETBENCH_HISTORY_MODE} history_action_chars=${SPREADSHEETBENCH_HISTORY_ACTION_CHARS} history_obs_chars=${SPREADSHEETBENCH_HISTORY_OBS_CHARS}"
echo "[spreadsheet-train-submit] python_error_penalty=${SPREADSHEETBENCH_PYTHON_ERROR_PENALTY} syntax_error_penalty=${SPREADSHEETBENCH_SYNTAX_ERROR_PENALTY}"
echo "[spreadsheet-train-submit] tool_set=${SPREADSHEETBENCH_TOOL_SET}"
echo "[spreadsheet-train-submit] actor_timeout_s=${SPREADSHEETBENCH_ACTOR_TIMEOUT_SECONDS} phase_heartbeat_s=${SPREADSHEETBENCH_PHASE_HEARTBEAT_SECONDS}"
echo "[spreadsheet-train-submit] submitting through ${RAY_DASHBOARD_ADDRESS}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  REDACTED_RUNTIME_ENV_JSON="$(
    RUNTIME_ENV_JSON_VALUE="${RUNTIME_ENV_JSON}" "${PY}" - <<'PY'
import json
import os

payload = json.loads(os.environ["RUNTIME_ENV_JSON_VALUE"])
if "WANDB_API_KEY" in payload.get("env_vars", {}):
    payload["env_vars"]["WANDB_API_KEY"] = "***"
print(json.dumps(payload))
PY
  )"
  echo "[spreadsheet-train-submit] runtime_env=${REDACTED_RUNTIME_ENV_JSON}"
  echo "[spreadsheet-train-submit] dry run complete"
  exit 0
fi
ray job submit \
  --address="${RAY_DASHBOARD_ADDRESS}" \
  --working-dir "${ROOT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- bash -lc \
  'cd "$ENVHARNESS_ROOT" && bash rl/scripts/run_spreadsheetbench_grpo.sh'
