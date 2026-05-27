#!/bin/bash

set -euo pipefail

# ===== 全流程开关与公共配置 =====
PROJ_ROOT="/public/home/users/hz3b-zk2/code/CResUNet"
CONDA_ENV_NAME=ai

# ============================================================================
# 选择数据管线: macom (新默认) / fvcom (旧)
# ============================================================================
# --- macom 模式 (新: 原始高分辨率 MaCOM + GLO12 低分辨率监督) ---
DATA_SOURCE=macom
SOURCE_ROOT="/public/home/users/shwadata/MODELDATA/CURRENT_MACOM_SH"
FORECAST_PATTERN="$SOURCE_ROOT/*/*_swt_SH_001h_*.nc"
# GLO12_PATTERN="/path/to/GLO12/*.nc"    # 如有 GLO12 数据请填写
USER_OUT="$PROJ_ROOT/out_macom"
MODEL_PATH="$PROJ_ROOT/train_results_macom/best_model.pth"

# --- fvcom 模式 (旧管线: 插值到 608×704 网格) ---
# 取消注释以下变量并注释掉上方 macom 块即可切换
# DATA_SOURCE=fvcom
# SOURCE_ROOT="/public/home/users/hz3b-zk2/code/data"
# FORECAST_PATTERN="$SOURCE_ROOT/sst_forecast/*.nc"
# REANALYSIS_PATTERN="$SOURCE_ROOT/sst_reanalysis/*.nc"
# USER_OUT="$PROJ_ROOT/out"
# MODEL_PATH="$PROJ_ROOT/train_results/best_model.pth"
# FORECAST_PATH="$PROJ_ROOT/data/forecast_structured.nc"
# OUTPUT_NC="$USER_OUT/forecast_corrected_structured.nc"
# SAVE_REANALYSIS=1

# ============================================================================
# 通用配置
# ============================================================================
OUTPUT_DIR="$PROJ_ROOT/data"
USER_IN="$PROJ_ROOT/data"
DEVICE=cuda
SAVE_BIAS=${SAVE_BIAS:-0}

# 日期筛选
START_DATE=${START_DATE:-"20260401"}
END_DATE=${END_DATE:-"20260501"}

# ============================================

export PROJ_ROOT
export CONDA_ENV_NAME
export DATA_SOURCE
export SOURCE_ROOT
export FORECAST_PATTERN
export OUTPUT_DIR
export USER_IN
export USER_OUT
export MODEL_PATH
export DEVICE
export START_DATE
export END_DATE
export SAVE_BIAS

# macom 模式
if [ "$DATA_SOURCE" = "macom" ]; then
    [ -n "${GLO12_PATTERN:-}" ] && export GLO12_PATTERN
    echo "[Macom 模式] 直接推理 ..."
    bash "$PROJ_ROOT/run_predict.sh"

# fvcom 模式
elif [ "$DATA_SOURCE" = "fvcom" ]; then
    export FORECAST_PATH
    export OUTPUT_NC
    export REANALYSIS_PATTERN
    export SAVE_REANALYSIS

    RUN_DATA_PROCESS=${RUN_DATA_PROCESS:-1}
    RUN_PREDICT=${RUN_PREDICT:-1}

    if [ "$RUN_DATA_PROCESS" = "1" ]; then
        echo "[FVcom 模式 Stage 1/2] Data processing ..."
        bash "$PROJ_ROOT/run_data_process.sh"
    fi

    if [ "$RUN_PREDICT" = "1" ]; then
        echo "[FVcom 模式 Stage 2/2] Prediction ..."
        bash "$PROJ_ROOT/run_predict.sh"
    fi

else
    echo "[run_all] 未知 DATA_SOURCE=$DATA_SOURCE，请设为 macom 或 fvcom。"
    exit 1
fi

echo "Pipeline finished."
