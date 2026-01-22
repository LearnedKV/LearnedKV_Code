#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compare Adaptive vs Fixed compression on a benchmark
#
# This script runs both fixed and adaptive compression with the same overall
# compression budget to compare their effectiveness.
#
# Usage:
#   ./evaluate_adaptive_vs_fixed.sh <policy_checkpoint> <c_target> <dataset> <data_dir> [base_press] [model]
#
# Examples:
#   ./evaluate_adaptive_vs_fixed.sh ckpts/policy/policy_epoch10.pt 0.5 ruler 4096
#   ./evaluate_adaptive_vs_fixed.sh ckpts/policy/policy_epoch10.pt 0.6 longbench null snapkv

set -e

# Parse arguments
POLICY_CKPT=${1:-"/data/evictionKV/ckpts/policy_qwen/policy_qwen_epoch10.pt"}
C_TARGET=${2:-0.5}
DATASET=${3:-"ruler"}
DATA_DIR=${4:-"4096"}
BASE_PRESS=${5:-"snapkv"}
# MODEL=${6:-"meta-llama/Llama-2-7b-hf"}
MODEL=${6:-"Qwen/Qwen2.5-7B-Instruct"}
# Calculate compression_ratio from c_target
COMPRESSION_RATIO=$(python3 -c "print(1.0 - $C_TARGET)")

echo "============================================"
echo "Adaptive vs Fixed Compression Comparison"
echo "============================================"
echo "Dataset: $DATASET"
echo "Data dir: $DATA_DIR"
echo "Model: $MODEL"
echo "Base press: $BASE_PRESS"
echo "Target retention (c_target): $C_TARGET"
echo "Compression ratio: $COMPRESSION_RATIO"
echo "============================================"
echo ""

# Check if policy checkpoint exists
if [ ! -f "$POLICY_CKPT" ]; then
    echo "ERROR: Policy checkpoint not found at: $POLICY_CKPT"
    exit 1
fi

# Baseline - No compression
echo "--------------------------------------------"
echo "1. Running BASELINE (no compression)..."
echo "--------------------------------------------"
cmd_baseline="python evaluate.py \
    --dataset $DATASET \
    --model $MODEL \
    --press_name no_press \
    --device cuda:7"

if [ "$DATA_DIR" != "null" ] && [ ! -z "$DATA_DIR" ]; then
    cmd_baseline="$cmd_baseline --data_dir $DATA_DIR"
fi

echo "Running: $cmd_baseline"
eval $cmd_baseline
echo ""

# Fixed compression
echo "--------------------------------------------"
echo "2. Running FIXED compression..."
echo "--------------------------------------------"
cmd_fixed="python evaluate.py \
    --dataset $DATASET \
    --model $MODEL \
    --press_name $BASE_PRESS \
    --compression_ratio $COMPRESSION_RATIO\
    --device cuda:7"

if [ "$DATA_DIR" != "null" ] && [ ! -z "$DATA_DIR" ]; then
    cmd_fixed="$cmd_fixed --data_dir $DATA_DIR"
fi

echo "Running: $cmd_fixed"
eval $cmd_fixed
echo ""

# Adaptive compression
echo "--------------------------------------------"
echo "3. Running ADAPTIVE compression..."
echo "--------------------------------------------"
ADAPTIVE_PRESS="adaptive_${BASE_PRESS}"

cmd_adaptive="python evaluate.py \
    --dataset $DATASET \
    --model $MODEL \
    --press_name $ADAPTIVE_PRESS \
    --policy_checkpoint $POLICY_CKPT \
    --c_target $C_TARGET \
    --min_tokens_per_head 1.0\
    --device cuda:7"

if [ "$DATA_DIR" != "null" ] && [ ! -z "$DATA_DIR" ]; then
    cmd_adaptive="$cmd_adaptive --data_dir $DATA_DIR"
fi

echo "Running: $cmd_adaptive"
eval $cmd_adaptive
echo ""

echo "============================================"
echo "All evaluations completed!"
echo "============================================"
echo ""
echo "Results summary:"
echo "  1. Baseline (no compression): results/*__no_press__*/"
echo "  2. Fixed compression: results/*__${BASE_PRESS}__${COMPRESSION_RATIO}__*/"
echo "  3. Adaptive compression: results/*__${ADAPTIVE_PRESS}__*__c_target${C_TARGET}__*/"
echo ""
echo "To compare metrics:"
echo "  cat results/*__no_press__*/metrics.json"
echo "  cat results/*__${BASE_PRESS}__*/metrics.json"
echo "  cat results/*__${ADAPTIVE_PRESS}__*/metrics.json"

