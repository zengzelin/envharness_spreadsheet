#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash rl/scripts/submit_spreadsheetbench_eval.sh LABEL=MODEL [LABEL=MODEL ...]

Each MODEL must be a Hugging Face directory containing config.json and model
weights. Evaluations are submitted sequentially to the existing Ray cluster.

Important overrides:
  VAL_SIZE=399          Number of validation tasks to evaluate.
  VAL_CONCURRENCY=64    Maximum validation Ray actors used at once.
  MAX_STEPS=15          Environment turns per task.
EOF
}

if (( $# == 0 )); then
  usage >&2
  exit 2
fi

SPREADSHEET_RL_DATA_ROOT="${SPREADSHEET_RL_DATA_ROOT:-${ROOT}/experiments/spreadsheetbench/data/Spreadsheet-RL}"
SPREADSHEET_RL_TRAIN_FILE="${SPREADSHEET_RL_TRAIN_FILE:-train_hermes.parquet}"
SPREADSHEET_RL_VAL_FILE="${SPREADSHEET_RL_VAL_FILE:-test_verified_hermes.parquet}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs}"
VAL_SIZE="${VAL_SIZE:-399}"
VAL_CONCURRENCY="${VAL_CONCURRENCY:-64}"
VAL_BS="${VAL_BS:-${VAL_SIZE}}"
MAX_STEPS="${MAX_STEPS:-15}"
RAY_STATE_FILE="${RAY_STATE_FILE:-${ROOT}/runs/ray/ray_address.env}"
if [[ -f "${RAY_STATE_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${RAY_STATE_FILE}"
fi
EVAL_TOTAL_GPUS=$(( ${NNODES:-1} * ${N_GPUS_PER_NODE:-${GPUS_PER_NODE:-8}} ))

for spec in "$@"; do
  if [[ "${spec}" != *=* ]]; then
    echo "model specification must be LABEL=MODEL, got: ${spec}" >&2
    exit 2
  fi
  label="${spec%%=*}"
  model="${spec#*=}"
  if [[ ! "${label}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid evaluation label: ${label}" >&2
    exit 2
  fi
  if [[ ! -d "${model}" ]]; then
    echo "model directory not found: ${model}" >&2
    exit 2
  fi
  model="$(cd "${model}" && pwd)"
  if [[ ! -f "${model}/config.json" ]]; then
    echo "model config not found: ${model}/config.json" >&2
    exit 2
  fi
  if ! compgen -G "${model}/*.safetensors" >/dev/null \
      && ! compgen -G "${model}/*.bin" >/dev/null; then
    echo "model weights not found in: ${model}" >&2
    exit 2
  fi

  run_ts="$(date +%Y%m%d_%H%M%S)"
  exp_name="spreadsheetrl_full${VAL_SIZE}_eval_${label}_${run_ts}"
  run_dir="${RUN_ROOT}/grpo_spreadsheetbench_eval_${label}_${run_ts}"
  echo "[spreadsheet-eval] label=${label} model=${model}"
  echo "[spreadsheet-eval] val_size=${VAL_SIZE} val_concurrency=${VAL_CONCURRENCY} run_dir=${run_dir}"

  MODE=diagnostic \
  MODEL="${model}" \
  RESUME_FROM="" \
  TRAIN_BS="${TRAIN_BS:-${EVAL_TOTAL_GPUS}}" \
  PPO_MINI_BS="${PPO_MINI_BS:-${EVAL_TOTAL_GPUS}}" \
  GROUP_N="${GROUP_N:-1}" \
  VAL_BS="${VAL_BS}" \
  VAL_SIZE="${VAL_SIZE}" \
  VAL_CONCURRENCY="${VAL_CONCURRENCY}" \
  EPOCHS=1 \
  TOTAL_TRAINING_STEPS=1 \
  VAL_BEFORE=True \
  EXTRA_HYDRA="trainer.val_only=True" \
  MAX_STEPS="${MAX_STEPS}" \
  HISTORY_LENGTH="${HISTORY_LENGTH:-2}" \
  MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}" \
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}" \
  APPLY_CHAT_TEMPLATE_ENABLE_THINKING="${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-True}" \
  ENVHARNESS_DISABLE_THINKING="${ENVHARNESS_DISABLE_THINKING:-0}" \
  VAL_TEMPERATURE="${VAL_TEMPERATURE:-0}" \
  VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}" \
  SPREADSHEETBENCH_DATA_FORMAT=spreadsheet_rl \
  SPREADSHEET_RL_DATA_ROOT="${SPREADSHEET_RL_DATA_ROOT}" \
  SPREADSHEET_RL_TRAIN_FILE="${SPREADSHEET_RL_TRAIN_FILE}" \
  SPREADSHEET_RL_VAL_FILE="${SPREADSHEET_RL_VAL_FILE}" \
  SPREADSHEETBENCH_TOOL_SET="${SPREADSHEETBENCH_TOOL_SET:-native_basic}" \
  SPREADSHEETBENCH_HISTORY_MODE="${SPREADSHEETBENCH_HISTORY_MODE:-compact}" \
  RUN_ROOT="${RUN_ROOT}" \
  RUN_DIR="${run_dir}" \
  RUN_TS="${run_ts}" \
  EXP_NAME="${exp_name}" \
  WANDB_NAME="${exp_name}" \
  WANDB_PROJECT="${WANDB_PROJECT:-envharness_rl_spreadsheetbench}" \
  bash "${SCRIPT_DIR}/submit_spreadsheetbench_grpo.sh"

  echo "[spreadsheet-eval] completed label=${label} run_dir=${run_dir}"
done
