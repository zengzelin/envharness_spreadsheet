#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fetch verl-agent into third_party/verl-agent and apply the EnvHarness routes.
#
# verl-agent is NOT vendored in this repository. This script reproduces the
# exact tree the RL experiments ran against:
#
#   1. clone https://github.com/langfengQ/verl-agent at the pinned commit
#   2. apply rl/integration/verl_agent_env_manager.patch -- additive
#      `envharness_rl/alfworld` and `envharness_rl/spreadsheetbench` routes. See
#      rl/integration/ENVHARNESS_CHANGES.md for the full modification notes.
#   3. apply rl/integration/verl_agent_tracking_lifecycle.patch -- explicit,
#      idempotent tracker shutdown for Ray-hosted training jobs.
#   4. apply rl/integration/verl_agent_spreadsheetbench_runtime.patch --
#      Spreadsheet-RL splits, structured diagnostics, and error metrics.
#   5. apply rl/integration/verl_agent_spreadsheetbench_stall_diagnostics.patch
#      -- phase heartbeats around environment and model rollout calls.
#   6. apply rl/integration/verl_agent_spreadsheetbench_metrics.patch --
#      native-tool metrics, weighted validation, and reward components.
#   7. apply rl/integration/verl_agent_spreadsheetbench_eval.patch -- avoid
#      allocating unused training environment actors for val-only jobs.
#
# The result lands at third_party/verl-agent/ (gitignored). Optional extras
# (DAPO / Qwen3-8B / webshop / SWE-Gym) live in
# rl/integration/verl_agent_all_changes.patch; apply it INSTEAD of the
# env_manager patch (it is a superset).
#
# Usage:
#   bash rl/scripts/fetch_verl_agent.sh            # default env_manager patch
#   PATCH=all bash rl/scripts/fetch_verl_agent.sh  # full adaptation set
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/third_party/verl-agent"
UPSTREAM="https://github.com/langfengQ/verl-agent"
COMMIT="796ed310287fa605c9292a0fce07a86d79fde05e"

case "${PATCH:-env_manager}" in
  all) PATCH_FILE="$ROOT/rl/integration/verl_agent_all_changes.patch" ;;
  *)   PATCH_FILE="$ROOT/rl/integration/verl_agent_env_manager.patch" ;;
esac
TRACKING_PATCH_FILE="$ROOT/rl/integration/verl_agent_tracking_lifecycle.patch"
SPREADSHEET_RUNTIME_PATCH_FILE="$ROOT/rl/integration/verl_agent_spreadsheetbench_runtime.patch"
SPREADSHEET_STALL_PATCH_FILE="$ROOT/rl/integration/verl_agent_spreadsheetbench_stall_diagnostics.patch"
SPREADSHEET_METRICS_PATCH_FILE="$ROOT/rl/integration/verl_agent_spreadsheetbench_metrics.patch"
SPREADSHEET_EVAL_PATCH_FILE="$ROOT/rl/integration/verl_agent_spreadsheetbench_eval.patch"

if [ -e "$DEST" ]; then
  HEAD_COMMIT="$(git -C "$DEST" rev-parse HEAD 2>/dev/null || true)"
  ROUTE_MARKER='envharness_rl/spreadsheetbench'
  if [ "${PATCH:-env_manager}" = "all" ]; then
    MODE_MARKER='envharness_swegym_openhands'
  else
    MODE_MARKER='envharness_rl/alfworld'
  fi
  if [ "$HEAD_COMMIT" = "$COMMIT" ] &&
     grep -q "$ROUTE_MARKER" "$DEST/agent_system/environments/env_manager.py" &&
     grep -q "$MODE_MARKER" "$DEST/agent_system/environments/env_manager.py" &&
     grep -q 'env_python_runtime_error' "$DEST/agent_system/multi_turn_rollout/rollout_loop.py" &&
     grep -q 'SPREADSHEETBENCH_PHASE_HEARTBEAT_SECONDS' "$DEST/agent_system/multi_turn_rollout/rollout_loop.py" &&
     grep -q 'episode_tool_calls_per_turn' "$DEST/agent_system/multi_turn_rollout/rollout_loop.py" &&
     grep -q 'env_multi_call_short_circuit' "$DEST/agent_system/multi_turn_rollout/rollout_loop.py" &&
     grep -q 'reward_workbook_score' "$DEST/verl/trainer/ppo/ray_trainer.py" &&
     grep -q 'success_rate_weights' "$DEST/verl/trainer/ppo/ray_trainer.py" &&
     grep -q '_val_only = bool(config.trainer.get("val_only", False))' "$DEST/agent_system/environments/env_manager.py" &&
     grep -q 'def finish' "$DEST/verl/utils/tracking.py"; then
    echo "[fetch_verl_agent] $DEST already at $COMMIT + patches; nothing to do."
    exit 0
  fi
  echo "[fetch_verl_agent] $DEST exists but is not upstream+patch." >&2
  echo "Remove it (or fix it up manually) and re-run." >&2
  exit 1
fi

echo "[fetch_verl_agent] cloning $UPSTREAM @ $COMMIT -> $DEST"
git init -q "$DEST"
git -C "$DEST" remote add origin "$UPSTREAM"
# GitHub allows fetching an arbitrary SHA directly; fall back to a full
# fetch if the shallow-by-SHA path is refused.
if ! git -C "$DEST" fetch -q --depth 1 origin "$COMMIT"; then
  echo "[fetch_verl_agent] shallow fetch refused; falling back to full fetch"
  git -C "$DEST" fetch -q origin
fi
git -C "$DEST" checkout -q "$COMMIT"

echo "[fetch_verl_agent] applying $(basename "$PATCH_FILE")"
git -C "$DEST" apply "$PATCH_FILE"
echo "[fetch_verl_agent] applying $(basename "$SPREADSHEET_RUNTIME_PATCH_FILE")"
git -C "$DEST" apply "$SPREADSHEET_RUNTIME_PATCH_FILE"
echo "[fetch_verl_agent] applying $(basename "$SPREADSHEET_STALL_PATCH_FILE")"
git -C "$DEST" apply "$SPREADSHEET_STALL_PATCH_FILE"
echo "[fetch_verl_agent] applying $(basename "$SPREADSHEET_METRICS_PATCH_FILE")"
git -C "$DEST" apply "$SPREADSHEET_METRICS_PATCH_FILE"
echo "[fetch_verl_agent] applying $(basename "$SPREADSHEET_EVAL_PATCH_FILE")"
git -C "$DEST" apply "$SPREADSHEET_EVAL_PATCH_FILE"
echo "[fetch_verl_agent] applying $(basename "$TRACKING_PATCH_FILE")"
git -C "$DEST" apply "$TRACKING_PATCH_FILE"

echo "[fetch_verl_agent] done: $DEST (upstream $COMMIT + integration patches)"
echo "  license: Apache-2.0 -- see $DEST/LICENSE and $DEST/Notice.txt"
