#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Batch evaluation script for AdaptivePerHeadPress on multiple benchmarks
# 
# Usage:
#   ./evaluate_adaptive_all.sh <policy_checkpoint> <c_target> <model_name> [base_press]
#
# Examples:
#   ./evaluate_adaptive_all.sh ckpts/policy/llama-3.1-8b/policy_epoch10.pt 0.5 meta-llama/Meta-Llama-3.1-8B-Instruct observed_attention
#   ./evaluate_adaptive_all.sh ckpts/policy/llama-3.1-8b/policy_epoch10.pt 0.6 meta-llama/Meta-Llama-3.1-8B-Instruct snapkv

set -e  # Exit on error

# Parse arguments
POLICY_CKPT=${1:-"ckpts/policy/policy_epoch10.pt"}
C_TARGET=${2:-0.5}
MODEL=${3:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
BASE_PRESS=${4:-"observed_attention"}

echo "============================================"
echo "Adaptive Per-Head Compression Evaluation"
echo "============================================"
echo "Policy checkpoint: $POLICY_CKPT"
echo "c_target: $C_TARGET"
echo "Model: $MODEL"
echo "Base press: $BASE_PRESS"
echo "============================================"
echo ""

# Check if policy checkpoint exists
if [ ! -f "$POLICY_CKPT" ]; then
    echo "ERROR: Policy checkpoint not found at: $POLICY_CKPT"
    echo "Please train a policy model first using:"
    echo "  python policy/train.py --kv-root <kv_data_dir> --save-dir <save_dir>"
    exit 1
fi

ADAPTIVE_PRESS="adaptive_${BASE_PRESS}"

echo "Using press: $ADAPTIVE_PRESS"
echo ""

# Function to run evaluation
run_eval() {
    local dataset=$1
    local data_dir=$2
    local extra_args=$3
    
    echo "--------------------------------------------"
    echo "Evaluating on: $dataset (data_dir: $data_dir)"
    echo "--------------------------------------------"
    
    cmd="python evaluate.py \
        --dataset $dataset \
        --model $MODEL \
        --press_name $ADAPTIVE_PRESS \
        --policy_checkpoint $POLICY_CKPT \
        --c_target $C_TARGET \
        --min_tokens_per_head 1.0"
    
    if [ ! -z "$data_dir" ]; then
        cmd="$cmd --data_dir $data_dir"
    fi
    
    if [ ! -z "$extra_args" ]; then
        cmd="$cmd $extra_args"
    fi
    
    echo "Running: $cmd"
    echo ""
    
    eval $cmd
    
    echo ""
    echo "Completed: $dataset (data_dir: $data_dir)"
    echo ""
}

# RULER - Multiple context lengths
echo "=========================================="
echo "1. RULER Benchmark"
echo "=========================================="
for data_dir in 4096 8192 16384 32768 65536 131072; do
    run_eval "ruler" "$data_dir" ""
done

# LongBench
echo "=========================================="
echo "2. LongBench"
echo "=========================================="
run_eval "longbench" "" ""

# LongBench-v2
echo "=========================================="
echo "3. LongBench-v2"
echo "=========================================="
run_eval "longbench-v2" "" ""

# InfiniteBench
echo "=========================================="
echo "4. InfiniteBench"
echo "=========================================="
run_eval "infinitebench" "" ""

# Loogle
echo "=========================================="
echo "5. Loogle"
echo "=========================================="
run_eval "loogle" "" ""

# Zero Scrolls
echo "=========================================="
echo "6. Zero Scrolls"
echo "=========================================="
run_eval "zero_scrolls" "" ""

# Needle in Haystack - Multiple depths
echo "=========================================="
echo "7. Needle in Haystack"
echo "=========================================="
for depth in 10 30 50 70 90; do
    run_eval "needle_in_haystack" "" "--needle_depth $depth --max_context_length 32768"
done

echo "============================================"
echo "All evaluations completed!"
echo "============================================"
echo ""
echo "Results are saved in: ./results/"
echo ""
echo "To analyze results, run:"
echo "  python analysis/analyze_results.py --results_dir ./results/"

