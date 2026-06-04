#!/bin/bash
set -euo pipefail

PROJ_ROOT="/public/home/users/hz3b-zk2/code/CResUNet"
LOG_DIR="$PROJ_ROOT/logs"
mkdir -p "$LOG_DIR"

TARGET_DATE=${TARGET_DATE:-$(date +%Y%m%d)}
LOG_FILE="$LOG_DIR/run_${TARGET_DATE}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "处理日期: $TARGET_DATE" | tee -a "$LOG_FILE"

export START_DATE="${TARGET_DATE}"
export END_DATE=$(date -d "${TARGET_DATE}+1 day" +%Y%m%d)
export USER_OUT="/public/home/users/shwadata/MODELDATA/AI_CResU-net_sstR_SH"

echo "--- 运行 run_all.sh ---" | tee -a "$LOG_FILE"
bash "$PROJ_ROOT/run_all.sh" >> "$LOG_FILE" 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ run_all.sh 失败，跳过 predict" | tee -a "$LOG_FILE"
    exit 1
fi

echo "✅ run_all.sh 完成" | tee -a "$LOG_FILE"

echo "--- 运行 predict_demo_macom.py ---" | tee -a "$LOG_FILE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ai

python "$PROJ_ROOT/predict_demo_macom.py" \
    --date "$TARGET_DATE" \
    --step 0 \
    --corrected-dir "$USER_OUT" \
    --outdir "$USER_OUT/demo" 2>&1 | tee -a "$LOG_FILE"

echo "✅ predict 完成" | tee -a "$LOG_FILE"
echo "结束时间: $(date)" | tee -a "$LOG_FILE"
