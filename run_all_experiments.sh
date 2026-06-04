#!/bin/bash
# ============================================================================
#  BSPC 论文实验一键启动脚本 (Linux Bash)
#  运行所有必要实验以生成论文结果数据
#  协议统一：Session 1 训练 → Session 2 测试（cross-session evaluation）
#
#  使用方法：
#    1. 把整个项目文件夹复制到 Linux 工作站
#    2. 确保安装了依赖（pip install -r requirements.txt）
#    3. 在项目根目录下运行：
#       bash run_all_experiments.sh
#
#  如果要在后台运行（关掉终端也不中断）：
#    nohup bash run_all_experiments.sh > run.log 2>&1 &
# ============================================================================

set -e  # 遇到错误即停止（如需忽略错误可改为 set +e）
# set +e  # 如果希望某个实验失败后继续跑后面的，取消这行注释，注释掉上一行

# --------------- Configuration ---------------
# 自动检测脚本所在目录（即项目根目录）
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results"
LOG_DIR="$RESULTS_DIR/run_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"

# 确保matplotlib在无显示器环境下也能保存图片
export MPLBACKEND=Agg

# Paper-relevant algorithms (including deep learning baselines)
PAPER_ALGORITHMS=(
    "CSP+LDA"
    "CSP+SVM"
    "FBCSP"
    "EEGNet"
    "ShallowFBCSPNet"
    "MDM"
    "RiemannTangentSpace+SVM"
    "FilterBankTangentSpace+SVM"
)

ALL_SUBJECTS=(1 2 3 4 5 6 7 8 9)

START_TIME=$(date +%s)

echo ""
echo "================================================"
echo "  BSPC Paper: All Experiments Runner"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Algorithms: ${PAPER_ALGORITHMS[*]}"
echo "  Deep Learning: EEGNet, ShallowFBCSPNet (300 epochs)"
echo "================================================"
echo ""

# Check GPU availability
echo ">>> Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU FOUND: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print('  Deep learning models will use GPU acceleration.')
else:
    print('  No GPU found. Deep learning models will train on CPU (may be slow).')
" 2>&1 | tee "$LOG_DIR/00_GPU_Check.log"
echo ""

# --------------- Helper function ---------------
run_cmd() {
    local name="$1"
    local cmd="$2"
    local log_file="$LOG_DIR/$(echo "$name" | sed 's/[^a-zA-Z0-9_]/_/g').log"

    echo ""
    echo ">>> $name"
    echo "------------------------------------------------------------"
    echo "  Command: $cmd"

    # Run command, tee to both log file and console
    eval "$cmd" 2>&1 | tee "$log_file"

    local exit_code="${PIPESTATUS[0]}"
    if [ "$exit_code" -ne 0 ]; then
        echo "  [WARNING] Command completed with exit code: $exit_code" >&2
    fi

    echo "  [DONE] Results saved to: $log_file"
    echo ""
}

# ===========================================================================
#  EXPERIMENT 1: Main Evaluation (BCI IV 2A)
#  Compare FBTS against baseline algorithms
#  Protocol: Session 1 → Session 2 (cross-session, no CV)
# ===========================================================================
run_cmd "01_Main_Evaluation_BCI_IV_2A" \
"cd \"$PROJECT_ROOT\" && python evaluate_algorithms.py --dataset BCI_IV_2A --subjects 1~9 --algorithms ${PAPER_ALGORITHMS[*]}"

# Capture the main evaluation results file for statistical analysis
# (Experiment 1b also generates evaluation_results_bciiv2a_*.csv, so we need to capture the correct one)
MAIN_EVAL_RESULTS=$(ls -t "$PROJECT_ROOT/results/evaluation_results_bciiv2a_"*.csv 2>/dev/null | head -1)
echo "Main evaluation results file: $MAIN_EVAL_RESULTS"
if [ -z "$MAIN_EVAL_RESULTS" ]; then
    echo "[WARNING] No evaluation results file found. Statistical analysis may fail."
fi

# ===========================================================================
#  EXPERIMENT 1b: t-SNE Visualization (FBTS+SVM)
#  Generate t-SNE feature visualization for FilterBankTangentSpace+SVM
#  Protocol: Session 1 → Session 2
# ===========================================================================
run_cmd "01b_tSNE_Visualization" \
"cd \"$PROJECT_ROOT\" && python evaluate_algorithms.py --subjects 1~9 --tsne --algorithms FilterBankTangentSpace+SVM --dataset BCI_IV_2A"

# ===========================================================================
#  EXPERIMENT 2: Ablation Study
#  Systematic component analysis across all 9 subjects
#  Protocol: Session 1 → Session 2 (hardcoded in script)
# ===========================================================================
run_cmd "02_Ablation_Study" \
"cd \"$PROJECT_ROOT\" && python experiments/ablation_study_all_subjects.py --subjects 1~9"

# ===========================================================================
#  EXPERIMENT 3: Channel Configuration Comparison
#  Evaluate different channel configurations (22ch vs 8ch vs 6ch vs 4ch)
#  Protocol: Session 1 → Session 2 (cross-session, no CV)
# ===========================================================================
run_cmd "03_Channel_Comparison" \
"cd \"$PROJECT_ROOT\" && python compare_channel_configs.py --subjects 1~9 --configs all_channels motor_core_8 motor_core_6 minimal_4"

# ===========================================================================
#  EXPERIMENT 4: Cross-Dataset Evaluation (PhysionetMI)
#  Test generalizability on independent dataset
#  Protocol: Session 1 → Session 2
# ===========================================================================
run_cmd "04_Cross_Dataset_PhysionetMI" \
"cd \"$PROJECT_ROOT\" && python evaluate_algorithms.py --dataset PhysionetMI --subjects 1~109 --algorithms FilterBankTangentSpace+SVM"

# ===========================================================================
#  EXPERIMENT 5: Statistical Analysis
#  Generate significance tests and reports
#  Uses the main evaluation results from Experiment 1 (NOT the t-SNE results)
# ===========================================================================
run_cmd "05_Statistical_Analysis" \
"cd \"$PROJECT_ROOT\" && python experiments/statistical_analysis.py --input \"$MAIN_EVAL_RESULTS\""

# ===========================================================================
#  SUMMARY
# ===========================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo ""
echo "================================================"
echo "  ALL EXPERIMENTS COMPLETE!"
echo "  Started: $(date -d @$START_TIME '+%Y-%m-%d %H:%M:%S')"
echo "  Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  Results: $LOG_DIR"
echo "================================================"
echo ""

# List result files
echo ">>> Generated Result Files"
echo "------------------------------------------------------------"
ls -lh "$RESULTS_DIR"/*.csv 2>/dev/null | head -20 | awk '{print "  " $6, $7, $8, $9, "(" $5 ")"}'

echo ""
echo "NOTE: To regenerate with different settings, edit this script and re-run:"
echo "  bash run_all_experiments.sh"
echo ""