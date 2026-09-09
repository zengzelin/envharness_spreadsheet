#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="$(cd "${RL_ROOT}/.." && pwd)"
VERL_AGENT="${VERL_AGENT:-${ROOT}/third_party/verl-agent}"
PY="${PY:-$(command -v python || command -v python3)}"
MODE="${MODE:-smoke}"
NNODES="${NNODES:-1}"
if [[ ! "${NNODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NNODES must be a positive integer, got ${NNODES}" >&2
  exit 2
fi

if [[ -z "${RAY_ADDRESS:-}" ]]; then
  echo "SpreadsheetBench training requires an initialized external Ray cluster." >&2
  echo "Start it with: bash rl/scripts/mpi_ray_up.sh" >&2
  exit 1
fi
if [[ ! -d "${VERL_AGENT}/verl" ]]; then
  echo "verl-agent not found at ${VERL_AGENT}" >&2
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

case "${MODE}" in
  smoke)
    TRAIN_BS="${TRAIN_BS:-$((2 * NNODES))}"
    VAL_BS="${VAL_BS:-$((2 * NNODES))}"
    GROUP_N="${GROUP_N:-2}"
    PPO_MINI_BS="${PPO_MINI_BS:-$((2 * NNODES))}"
    EPOCHS="${EPOCHS:-1}"
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
    TP="${TP:-2}"
    MAX_STEPS="${MAX_STEPS:-3}"
    TEST_FREQ="${TEST_FREQ:-1}"
    VAL_BEFORE="${VAL_BEFORE:-False}"
    SAVE_FREQ="${SAVE_FREQ:-1}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
    ;;
  full)
    TRAIN_BS="${TRAIN_BS:-$((16 * NNODES))}"
    VAL_BS="${VAL_BS:-$((32 * NNODES))}"
    GROUP_N="${GROUP_N:-8}"
    PPO_MINI_BS="${PPO_MINI_BS:-$((16 * NNODES))}"
    EPOCHS="${EPOCHS:-150}"
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
    TP="${TP:-2}"
    MAX_STEPS="${MAX_STEPS:-15}"
    TEST_FREQ="${TEST_FREQ:-10}"
    VAL_BEFORE="${VAL_BEFORE:-True}"
    SAVE_FREQ="${SAVE_FREQ:-10}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
    ;;
  diagnostic)
    TRAIN_BS="${TRAIN_BS:-$((16 * NNODES))}"
    VAL_BS="${VAL_BS:-$((32 * NNODES))}"
    GROUP_N="${GROUP_N:-8}"
    PPO_MINI_BS="${PPO_MINI_BS:-$((16 * NNODES))}"
    EPOCHS="${EPOCHS:-20}"
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
    TP="${TP:-2}"
    MAX_STEPS="${MAX_STEPS:-10}"
    TEST_FREQ="${TEST_FREQ:-5}"
    VAL_BEFORE="${VAL_BEFORE:-True}"
    SAVE_FREQ="${SAVE_FREQ:-10}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
    ;;
  *)
    echo "MODE must be smoke, full, or diagnostic" >&2
    exit 2
    ;;
esac

HISTORY_LENGTH="${HISTORY_LENGTH:-2}"
if [[ "${MODE}" == "full" || "${MODE}" == "diagnostic" ]]; then
  DEFAULT_MODEL="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/luban/common/models/Qwen3-4B-Thinking-2507"
else
  DEFAULT_MODEL="${ROOT}/../llm_model/Qwen2.5-1.5B-Instruct"
  if [[ ! -f "${DEFAULT_MODEL}/config.json" ]]; then
    DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
  fi
fi
MODEL="${MODEL:-${DEFAULT_MODEL}}"
ENGINE="${ENGINE:-vllm}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
if [[ "${MODE}" == "full" ]]; then
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-24576}"
  APPLY_CHAT_TEMPLATE_ENABLE_THINKING="${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-True}"
  ENVHARNESS_DISABLE_THINKING="${ENVHARNESS_DISABLE_THINKING:-0}"
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
  ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}"
  ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-32768}"
  ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}"
  ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
  LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}"
elif [[ "${MODE}" == "diagnostic" ]]; then
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
  APPLY_CHAT_TEMPLATE_ENABLE_THINKING="${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-True}"
  ENVHARNESS_DISABLE_THINKING="${ENVHARNESS_DISABLE_THINKING:-0}"
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
  ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}"
  ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-24576}"
  ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}"
  ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
  LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-49152}"
else
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
  APPLY_CHAT_TEMPLATE_ENABLE_THINKING="${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-False}"
  ENVHARNESS_DISABLE_THINKING="${ENVHARNESS_DISABLE_THINKING:-1}"
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
  ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}"
  ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-}"
  ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-}"
  ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-}"
  LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-}"
fi
NUM_CPUS_PER_ENV="${NUM_CPUS_PER_ENV:-0.2}"
PPO_MICRO_BS_PER_GPU="${PPO_MICRO_BS_PER_GPU:-1}"
LOG_PROB_MICRO_BS_PER_GPU="${LOG_PROB_MICRO_BS_PER_GPU:-1}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.6}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs}"
VERL_DATA_DIR="${VERL_DATA_DIR:-${RUN_ROOT}/data/spreadsheetbench_agent/text}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/grpo_spreadsheetbench_${MODE}_${RUN_TS}}"
EXP_NAME="${EXP_NAME:-grpo_spreadsheetbench_${MODE}_${RUN_TS}}"
WANDB_PROJECT="${WANDB_PROJECT:-envharness_rl_spreadsheetbench}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://wandb.lubanml.woa.com}"
WANDB_MODE="${WANDB_MODE:-online}"
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
TRAINER_LOGGER="${TRAINER_LOGGER:-['console','wandb','tensorboard']}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-2}"

for integer_name in \
  TRAIN_BS VAL_BS GROUP_N PPO_MINI_BS EPOCHS N_GPUS_PER_NODE TP MAX_STEPS; do
  integer_value="${!integer_name}"
  if [[ ! "${integer_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${integer_name} must be a positive integer, got ${integer_value}" >&2
    exit 2
  fi
done
if [[ -n "${TOTAL_TRAINING_STEPS}" && ! "${TOTAL_TRAINING_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TOTAL_TRAINING_STEPS must be a positive integer, got ${TOTAL_TRAINING_STEPS}" >&2
  exit 2
fi
TOTAL_GPUS=$((NNODES * N_GPUS_PER_NODE))
if (( TRAIN_BS % TOTAL_GPUS != 0 )); then
  echo "TRAIN_BS must be divisible by total GPUs: ${TRAIN_BS} % ${TOTAL_GPUS} != 0" >&2
  exit 2
fi
if (( PPO_MINI_BS % TOTAL_GPUS != 0 )); then
  echo "PPO_MINI_BS must be divisible by total GPUs: ${PPO_MINI_BS} % ${TOTAL_GPUS} != 0" >&2
  exit 2
fi
if (( TP > N_GPUS_PER_NODE || N_GPUS_PER_NODE % TP != 0 )); then
  echo "TP must not exceed and must divide N_GPUS_PER_NODE: TP=${TP}, N_GPUS_PER_NODE=${N_GPUS_PER_NODE}" >&2
  exit 2
fi
if [[ -n "${GPUS_PER_NODE:-}" ]] && (( N_GPUS_PER_NODE > GPUS_PER_NODE )); then
  echo "N_GPUS_PER_NODE=${N_GPUS_PER_NODE} exceeds Ray node capacity ${GPUS_PER_NODE}" >&2
  exit 2
fi

mkdir -p \
  "${RUN_DIR}" "${VERL_DATA_DIR}" "${WANDB_DIR}" "${TENSORBOARD_DIR}" \
  "${ROLLOUT_DATA_DIR}" "${SPREADSHEETBENCH_TRAJECTORY_DIR}"
ln -sfn "${RUN_DIR}" "${RUN_ROOT}/grpo_spreadsheetbench_${MODE}_latest"

export SPREADSHEETBENCH_DATA
export SPREADSHEETBENCH_DATA_FORMAT
export SPREADSHEET_RL_DATA_ROOT="${SPREADSHEET_RL_DATA_ROOT:-}"
export SPREADSHEET_RL_TRAIN_FILE="${SPREADSHEET_RL_TRAIN_FILE:-}"
export SPREADSHEET_RL_VAL_FILE="${SPREADSHEET_RL_VAL_FILE:-}"
export SPREADSHEETBENCH_TRAJECTORY_DIR
export SPREADSHEETBENCH_HISTORY_MODE
export SPREADSHEETBENCH_HISTORY_ACTION_CHARS
export SPREADSHEETBENCH_HISTORY_OBS_CHARS
export SPREADSHEETBENCH_PYTHON_ERROR_PENALTY
export SPREADSHEETBENCH_SYNTAX_ERROR_PENALTY
export ENVHARNESS_ROOT="${ROOT}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export ENVHARNESS_DISABLE_THINKING
export PYTHONPATH="${ROOT}:${RL_ROOT}:${VERL_AGENT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export WANDB_BASE_URL WANDB_MODE WANDB_DIR WANDB_NAME TENSORBOARD_DIR
export MODE MODEL NNODES N_GPUS_PER_NODE TP TRAIN_BS VAL_BS GROUP_N
export PPO_MINI_BS EPOCHS MAX_STEPS HISTORY_LENGTH MAX_PROMPT_LENGTH
export MAX_RESPONSE_LENGTH APPLY_CHAT_TEMPLATE_ENABLE_THINKING ROLLOUT_DATA_DIR TRAINER_LOGGER WANDB_PROJECT
export RUN_DIR RUN_TS EXP_NAME TOTAL_TRAINING_STEPS

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  if [[ "${MODEL}" == /* && ! -f "${MODEL}/config.json" ]]; then
    echo "Model config not found: ${MODEL}/config.json" >&2
    exit 1
  fi
  command -v soffice >/dev/null 2>&1 || {
    echo "soffice is required for SpreadsheetBench grading" >&2
    exit 1
  }
  if [[ "${TRAINER_LOGGER}" == *wandb* ]]; then
    "${PY}" -c 'import wandb'
  fi
  if [[ "${TRAINER_LOGGER}" == *tensorboard* ]]; then
    "${PY}" -c 'from torch.utils.tensorboard import SummaryWriter'
  fi
  "${PY}" "${SCRIPT_DIR}/prepare_spreadsheetbench_verl_data.py" \
    --output-dir "${VERL_DATA_DIR}" \
    --train-size "${TRAIN_BS}" \
    --val-size "${VAL_BS}"
fi

CMD=(
  "${PY}" -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  "data.train_files=${VERL_DATA_DIR}/train.parquet"
  "data.val_files=${VERL_DATA_DIR}/test.parquet"
  "data.train_batch_size=${TRAIN_BS}"
  "data.val_batch_size=${VAL_BS}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "+data.apply_chat_template_kwargs.enable_thinking=${APPLY_CHAT_TEMPLATE_ENABLE_THINKING}"
  data.filter_overlong_prompts=True
  data.truncation=left
  data.return_raw_chat=True
  "actor_rollout_ref.model.path=${MODEL}"
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.model.use_remove_padding=True
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BS_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss=True
  "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BS_PER_GPU}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP}"
  "actor_rollout_ref.rollout.name=${ENGINE}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
  "actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K}"
  "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}"
  "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
  "actor_rollout_ref.rollout.enable_chunked_prefill=${ROLLOUT_ENABLE_CHUNKED_PREFILL}"
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=False
  "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BS_PER_GPU}"
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.actor.use_invalid_action_penalty=True
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
  algorithm.use_kl_in_reward=False
  env.env_name=envharness_rl/spreadsheetbench
  env.seed=0
  "env.history_length=${HISTORY_LENGTH}"
  "env.max_steps=${MAX_STEPS}"
  "env.rollout.n=${GROUP_N}"
  "env.resources_per_worker.num_cpus=${NUM_CPUS_PER_ENV}"
  env.resources_per_worker.num_gpus=0
  trainer.critic_warmup=0
  "trainer.logger=${TRAINER_LOGGER}"
  "trainer.project_name=${WANDB_PROJECT}"
  "trainer.experiment_name=${WANDB_NAME}"
  "trainer.log_val_generations=${LOG_VAL_GENERATIONS}"
  "trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}"
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.nnodes=${NNODES}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.default_local_dir=${RUN_DIR}/ckpts"
  "actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model']"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.total_epochs=${EPOCHS}"
  "trainer.val_before_train=${VAL_BEFORE}"
  +ray_init.address=auto
)

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  CMD+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi
if [[ -n "${ROLLOUT_MAX_MODEL_LEN}" ]]; then
  CMD+=("actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}")
fi
if [[ -n "${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" ]]; then
  CMD+=("actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}" ]]; then
  CMD+=("actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}")
fi
if [[ -n "${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" ]]; then
  CMD+=("actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}")
  CMD+=("actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}")
fi

if [[ -n "${RESUME_FROM:-}" ]]; then
  [[ -e "${RESUME_FROM}" ]] || { echo "RESUME_FROM not found: ${RESUME_FROM}" >&2; exit 1; }
  CMD+=(trainer.resume_mode=resume_path "trainer.resume_from_path=${RESUME_FROM}")
fi
if [[ -n "${EXTRA_HYDRA:-}" ]]; then
  # EXTRA_HYDRA is intentionally split into individual Hydra overrides.
  read -r -a EXTRA_ARGS <<< "${EXTRA_HYDRA}"
  CMD+=("${EXTRA_ARGS[@]}")
fi
CMD+=("$@")

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  MANIFEST_ENV="${RUN_DIR}/run_manifest.json" "${PY}" - "${CMD[@]}" <<'PY'
import json
import os
import subprocess
import sys

keys = [
    "MODE", "MODEL", "NNODES", "N_GPUS_PER_NODE", "TP", "TRAIN_BS",
    "VAL_BS", "GROUP_N", "PPO_MINI_BS", "EPOCHS", "TOTAL_TRAINING_STEPS",
    "MAX_STEPS", "TEST_FREQ", "SAVE_FREQ", "VAL_BEFORE",
    "HISTORY_LENGTH", "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH",
    "SPREADSHEETBENCH_DATA", "SPREADSHEETBENCH_DATA_FORMAT",
    "SPREADSHEET_RL_DATA_ROOT", "SPREADSHEET_RL_TRAIN_FILE",
    "SPREADSHEET_RL_VAL_FILE", "SPREADSHEETBENCH_TRAJECTORY_DIR",
    "SPREADSHEETBENCH_HISTORY_MODE", "SPREADSHEETBENCH_HISTORY_ACTION_CHARS",
    "SPREADSHEETBENCH_HISTORY_OBS_CHARS",
    "SPREADSHEETBENCH_PYTHON_ERROR_PENALTY",
    "SPREADSHEETBENCH_SYNTAX_ERROR_PENALTY",
    "WANDB_BASE_URL", "WANDB_MODE", "WANDB_DIR", "WANDB_NAME",
    "TENSORBOARD_DIR", "ROLLOUT_DATA_DIR", "TRAINER_LOGGER", "RUN_DIR",
    "RUN_TS", "EXP_NAME", "WANDB_PROJECT", "RAY_ADDRESS",
]
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=os.environ["ENVHARNESS_ROOT"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    git_commit = "unknown"
payload = {
    "git_commit": git_commit,
    "environment": {key: os.environ.get(key) for key in keys},
    "command": sys.argv[1:],
}
with open(os.environ["MANIFEST_ENV"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
PY
fi

{
  echo "[spreadsheet-train] mode=${MODE} model=${MODEL}"
  echo "[spreadsheet-train] ray=${RAY_ADDRESS} nodes=${NNODES} gpus_per_node=${N_GPUS_PER_NODE} tp=${TP}"
  echo "[spreadsheet-train] train_bs=${TRAIN_BS} group_n=${GROUP_N} epochs=${EPOCHS} total_training_steps=${TOTAL_TRAINING_STEPS:-auto} max_steps=${MAX_STEPS}"
  echo "[spreadsheet-train] eval_freq=${TEST_FREQ} save_freq=${SAVE_FREQ} val_before_train=${VAL_BEFORE}"
  echo "[spreadsheet-train] data_format=${SPREADSHEETBENCH_DATA_FORMAT} data=${SPREADSHEETBENCH_DATA}"
  if [[ "${SPREADSHEETBENCH_DATA_FORMAT}" == "spreadsheet_rl" || "${SPREADSHEETBENCH_DATA_FORMAT}" == "spreadsheet-rl" || "${SPREADSHEETBENCH_DATA_FORMAT}" == "spreadsheetrl" ]]; then
    echo "[spreadsheet-train] train_split=${SPREADSHEET_RL_TRAIN_FILE} val_split=${SPREADSHEET_RL_VAL_FILE}"
  fi
  echo "[spreadsheet-train] run_dir=${RUN_DIR}"
  echo "[spreadsheet-train] logger=${TRAINER_LOGGER} wandb_mode=${WANDB_MODE} wandb_name=${WANDB_NAME}"
  echo "[spreadsheet-train] history_mode=${SPREADSHEETBENCH_HISTORY_MODE} history_action_chars=${SPREADSHEETBENCH_HISTORY_ACTION_CHARS} history_obs_chars=${SPREADSHEETBENCH_HISTORY_OBS_CHARS}"
  echo "[spreadsheet-train] python_error_penalty=${SPREADSHEETBENCH_PYTHON_ERROR_PENALTY} syntax_error_penalty=${SPREADSHEETBENCH_SYNTAX_ERROR_PENALTY}"
  echo "[spreadsheet-train] rollouts=${ROLLOUT_DATA_DIR} trajectories=${SPREADSHEETBENCH_TRAJECTORY_DIR}"
  printf '[spreadsheet-train] command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
} | tee "${RUN_DIR}/launch.log"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[spreadsheet-train] dry run complete"
  exit 0
fi

cd "${VERL_AGENT}"
"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
