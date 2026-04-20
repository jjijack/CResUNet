#!/bin/bash

set -euo pipefail

# ===== 可配置部分 =====
PROJ_ROOT=${PROJ_ROOT:-"/home/user3/scratch/SST Correction/CResUNet"}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-torch}   # 设为空字符串则使用当前 python

# 原始数据目录
SOURCE_ROOT=${SOURCE_ROOT:-"/home/user3/scratch/SST Correction/Getea"}
FORECAST_PATTERN=${FORECAST_PATTERN:-"$SOURCE_ROOT/sst_forecast/*.nc"}
REANALYSIS_PATTERN=${REANALYSIS_PATTERN:-"$SOURCE_ROOT/sst_reanalysis/*.nc"}

# 输出目录
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJ_ROOT/data"}

# 默认仅生成 predict 所需的 forecast_structured.nc
SAVE_REANALYSIS=${SAVE_REANALYSIS:-0}

# 日期筛选（可空，为空表示全量）
START_DATE=${START_DATE:-""}   # 例如: 2023-01-01 或 20230101
END_DATE=${END_DATE:-""}       # 例如: 2023-03-31 或 20230331
# ============================================

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [ "$SAVE_REANALYSIS" = "1" ]; then
    EXTRA_ARGS+=(--save-reanalysis)
fi
if [ -n "$START_DATE" ]; then
    EXTRA_ARGS+=(--start-date "$START_DATE")
fi
if [ -n "$END_DATE" ]; then
    EXTRA_ARGS+=(--end-date "$END_DATE")
fi

if [ -n "$CONDA_ENV_NAME" ]; then
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV_NAME" python "$PROJ_ROOT/data/data_process.py" \
        --forecast-pattern "$FORECAST_PATTERN" \
        --reanalysis-pattern "$REANALYSIS_PATTERN" \
        --output-dir "$OUTPUT_DIR" \
        "${EXTRA_ARGS[@]}"
else
    PYTHONUNBUFFERED=1 python "$PROJ_ROOT/data/data_process.py" \
        --forecast-pattern "$FORECAST_PATTERN" \
        --reanalysis-pattern "$REANALYSIS_PATTERN" \
        --output-dir "$OUTPUT_DIR" \
        "${EXTRA_ARGS[@]}"
fi
