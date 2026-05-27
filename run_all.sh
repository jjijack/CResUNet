#!/bin/bash

set -euo pipefail

# ===== 全流程开关与公共配置 =====
PROJ_ROOT="/home/user3/scratch/SST Correction/CResUNet"
CONDA_ENV_NAME=torch

# --- fvcom 模式 ---
# DATA_SOURCE=fvcom
# SOURCE_ROOT="/home/user3/scratch/SST Correction/Getea"
# FORECAST_PATTERN="$SOURCE_ROOT/sst_forecast/*.nc"
# REANALYSIS_PATTERN="$SOURCE_ROOT/sst_reanalysis/*.nc"

# --- macom_compat 模式 (legacy: 插值到 608x704 的旧管线) ---
# DATA_SOURCE=macom_compat
# SOURCE_ROOT="/home/user3/scratch/SST Correction/CURRENT_MACOM_SH"
# FORECAST_PATTERN="$SOURCE_ROOT/*/*_swt_SH_001h_*.nc"
# REANALYSIS_PATTERN="/home/user3/scratch/SST Correction/Getea/sst_reanalysis/*.nc"

# --- macom 模式 (新默认: 原始高分辨率 + GLO12 低分辨率监督) ---
DATA_SOURCE=macom
SOURCE_ROOT="/home/user3/scratch/SST Correction/CURRENT_MACOM_SH"
FORECAST_PATTERN="$SOURCE_ROOT/*/*_swt_SH_001h_*.nc"
GLO12_PATTERN="/home/user3/scratch/SST Correction/GLO12/*.nc"

OUTPUT_DIR="$PROJ_ROOT/data"

# 推理输入输出路径（macom 新管线使用专用目录）
USER_IN="$PROJ_ROOT/data"
USER_OUT="$PROJ_ROOT/out_macom"
MODEL_PATH="$PROJ_ROOT/train_results_macom/best_model.pth"
DEVICE=cuda

# macom 新管线：predict 直接读原始 MaCOM 文件，输出到 out_macom/
# macom_compat/fvcom：predict 读 forecast_structured.nc，输出到 out/
FORECAST_PATH="$PROJ_ROOT/data/forecast_structured.nc"
OUTPUT_NC="$USER_OUT/forecast_corrected_structured.nc"

# 日期筛选会同时传给数据处理和预测
# fvcom 数据: START_DATE="20230301" END_DATE="20230401"
# macom 数据: START_DATE="20260401" END_DATE="20260501"
START_DATE=${START_DATE:-"20260401"}
END_DATE=${END_DATE:-"20260501"}

# 阶段开关（支持外部覆盖，如 RUN_PREDICT=0 bash run_all.sh）
RUN_DATA_PROCESS=${RUN_DATA_PROCESS:-1}
RUN_PREDICT=${RUN_PREDICT:-1}

# 数据处理阶段可选项
SAVE_REANALYSIS=0

# 预测阶段可选项
SAVE_BIAS=0
# ============================================

export PROJ_ROOT
export CONDA_ENV_NAME
export DATA_SOURCE
export SOURCE_ROOT
export FORECAST_PATTERN
export GLO12_PATTERN
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

# macom 新管线：跳过 run_data_process（训练时直接读原始文件）
if [ "$DATA_SOURCE" = "macom" ]; then
    echo "[Info] DATA_SOURCE=macom (新默认): 训练时直接读原始 MaCOM + GLO12 文件，跳过数据处理阶段。"
    echo "[Info] 如需使用旧管线（插值到 608x704），请设置 DATA_SOURCE=macom_compat"
    RUN_DATA_PROCESS=0
fi

if [ "$RUN_DATA_PROCESS" = "1" ]; then
    echo "[Stage 1/2] Data processing ..."
    bash "$PROJ_ROOT/run_data_process.sh"
fi

if [ "$RUN_PREDICT" = "1" ]; then
    echo "[Stage 2/2] Prediction ..."
    bash "$PROJ_ROOT/run_predict.sh"
fi

echo "Pipeline finished."
