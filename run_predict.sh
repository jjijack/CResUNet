#!/bin/bash

set -euo pipefail

# ===== 可配置部分 =====
PROJ_ROOT=${PROJ_ROOT:-"/home/user3/scratch/SST Correction/CResUNet"}
USER_IN=${USER_IN:-"$PROJ_ROOT/data"}
USER_OUT=${USER_OUT:-"$PROJ_ROOT/out"}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-torch}   # 设为空字符串则使用当前 python
DATA_SOURCE=${DATA_SOURCE:-""}

MODEL_PATH=${MODEL_PATH:-$PROJ_ROOT/train_results_macom/best_model.pth}
FORECAST_PATH=${FORECAST_PATH:-$PROJ_ROOT/data/forecast_structured.nc}
FORECAST_PATTERN=${FORECAST_PATTERN:-""}
DEVICE=${DEVICE:-cuda}
OUTPUT_NC=${OUTPUT_NC:-$USER_OUT/forecast_corrected_structured.nc}
SAVE_BIAS=${SAVE_BIAS:-0}  # 是否保存 bias 字段（forecast - corrected）
START_DATE=${START_DATE:-""}   # 例如: 2023-01-01 或 20230101
END_DATE=${END_DATE:-""}       # 例如: 2023-04-01 或 20230401 (含起始，不含结束)
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

if [ "$DATA_SOURCE" = "macom" ]; then
    # MaCOM 新管线：全图推理，输出高分辨率订正 SST
    if [ -n "${CONDA_ENV_NAME}" ]; then
        PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV_NAME" python "$PROJ_ROOT/predict_macom.py" \
            --model "$MODEL_PATH" \
            --forecast-pattern "$FORECAST_PATTERN" \
            --output-dir "$USER_OUT" \
            --device "$DEVICE" \
            "${EXTRA_ARGS[@]}"
    else
        PYTHONUNBUFFERED=1 python "$PROJ_ROOT/predict_macom.py" \
            --model "$MODEL_PATH" \
            --forecast-pattern "$FORECAST_PATTERN" \
            --output-dir "$USER_OUT" \
            --device "$DEVICE" \
            "${EXTRA_ARGS[@]}"
    fi
elif [ "$DATA_SOURCE" = "fvcom" ]; then
    # FVCOM 旧管线：读 forecast_structured.nc，输出 NetCDF
    if [ -n "${CONDA_ENV_NAME}" ]; then
        PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV_NAME" python "$PROJ_ROOT/predict_fvcom.py" \
            --model "$MODEL_PATH" \
            --forecast "$FORECAST_PATH" \
            --device "$DEVICE" \
            --output-nc "$OUTPUT_NC" \
            "${EXTRA_ARGS[@]}"
    else
        PYTHONUNBUFFERED=1 python "$PROJ_ROOT/predict_fvcom.py" \
            --model "$MODEL_PATH" \
            --forecast "$FORECAST_PATH" \
            --device "$DEVICE" \
            --output-nc "$OUTPUT_NC" \
            "${EXTRA_ARGS[@]}"
    fi
else
    echo "[Predict] 未知 DATA_SOURCE=$DATA_SOURCE，请设为 macom 或 fvcom。"
    exit 1
fi
