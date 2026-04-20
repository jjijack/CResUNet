#!/bin/bash

set -euo pipefail

# ===== 全流程开关与公共配置 =====
PROJ_ROOT="/home/user3/scratch/SST Correction/CResUNet"
CONDA_ENV_NAME=torch

# 数据处理输入输出路径
SOURCE_ROOT="/home/user3/scratch/SST Correction/Getea"
FORECAST_PATTERN="$SOURCE_ROOT/sst_forecast/*.nc"
REANALYSIS_PATTERN="$SOURCE_ROOT/sst_reanalysis/*.nc"
OUTPUT_DIR="$PROJ_ROOT/data"

# 推理输入输出路径
USER_IN="$PROJ_ROOT/data"
USER_OUT="$PROJ_ROOT/out"
MODEL_PATH="$PROJ_ROOT/train_results/best_model.pth"
FORECAST_PATH="$PROJ_ROOT/data/forecast_structured.nc"
DEVICE=cuda
OUTPUT_NC="$USER_OUT/forecast_corrected_structured.nc"

# 日期筛选会同时传给数据处理和预测
START_DATE="20230301"   # 例如: 2023-01-01 或 20230101
END_DATE="20230331"     # 例如: 2023-03-31 或 20230331

# 阶段开关
RUN_DATA_PROCESS=1
RUN_PREDICT=1

# 数据处理阶段可选项
SAVE_REANALYSIS=0

# 预测阶段可选项
SAVE_BIAS=0
# ============================================

export PROJ_ROOT
export CONDA_ENV_NAME
export SOURCE_ROOT
export FORECAST_PATTERN
export REANALYSIS_PATTERN
export OUTPUT_DIR
export USER_IN
export USER_OUT
export MODEL_PATH
export FORECAST_PATH
export DEVICE
export OUTPUT_NC
export START_DATE
export END_DATE
export SAVE_REANALYSIS
export SAVE_BIAS

if [ "$RUN_DATA_PROCESS" = "1" ]; then
    echo "[Stage 1/2] Data processing ..."
    bash "$PROJ_ROOT/run_data_process.sh"
fi

if [ "$RUN_PREDICT" = "1" ]; then
    echo "[Stage 2/2] Prediction ..."
    bash "$PROJ_ROOT/run_predict.sh"
fi

echo "Pipeline finished."
