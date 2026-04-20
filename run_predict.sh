#!/bin/bash

set -euo pipefail

# ===== 可配置部分 =====
PROJ_ROOT=${PROJ_ROOT:-"/home/user3/scratch/SST Correction/CResUNet"}
USER_IN=${USER_IN:-"$PROJ_ROOT/data"}
USER_OUT=${USER_OUT:-"$PROJ_ROOT/out"}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-torch}   # 设为空字符串则使用当前 python

MODEL_PATH=${MODEL_PATH:-$PROJ_ROOT/train_results/best_model.pth}
FORECAST_PATH=${FORECAST_PATH:-$PROJ_ROOT/data/forecast_structured.nc}
DEVICE=${DEVICE:-cuda}
OUTPUT_NC=${OUTPUT_NC:-$USER_OUT/forecast_corrected_structured.nc}
SAVE_BIAS=${SAVE_BIAS:-0}  # 是否保存 bias 字段（forecast - corrected）
START_DATE=${START_DATE:-""}   # 例如: 2023-01-01 或 20230101
END_DATE=${END_DATE:-""}       # 例如: 2023-03-31 或 20230331
# ============================================

mkdir -p "$(dirname "$OUTPUT_NC")"

EXTRA_ARGS=()
if [ "$SAVE_BIAS" = "1" ]; then
    EXTRA_ARGS+=(--save-bias)
fi
if [ -n "$START_DATE" ]; then
    EXTRA_ARGS+=(--start-date "$START_DATE")
fi
if [ -n "$END_DATE" ]; then
    EXTRA_ARGS+=(--end-date "$END_DATE")
fi

if [ -n "${CONDA_ENV_NAME}" ]; then
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV_NAME" python "$PROJ_ROOT/predict.py" \
        --model "$MODEL_PATH" \
        --forecast "$FORECAST_PATH" \
        --device "$DEVICE" \
        --output-nc "$OUTPUT_NC" \
        "${EXTRA_ARGS[@]}"
else
    PYTHONUNBUFFERED=1 python "$PROJ_ROOT/predict.py" \
        --model "$MODEL_PATH" \
        --forecast "$FORECAST_PATH" \
        --device "$DEVICE" \
        --output-nc "$OUTPUT_NC" \
        "${EXTRA_ARGS[@]}"
fi
