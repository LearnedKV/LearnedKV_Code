#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# Parallel Evaluation Script for Qwen3-8B on LongBench
# ============================================================================
# This script evaluates 7 ScorerPress methods (excluding snapkv and observed_attention)
# on Qwen3-8B model using the LongBench dataset with 4 compression ratios (0.1, 0.2, 0.3, 0.5)
# Each GPU (0-6) runs one method with all compression ratios sequentially
#
# Usage: ./evaluate_qwen3_longbench_parallel.sh
# ============================================================================

set -e

# Configuration
MODEL="Qwen/Qwen3-8B"
DATASET="longbench"

# All available LongBench tasks (verified from cache)
# Total: 35 tasks (34 actual tasks + repobench)
LONGBENCH_TASKS=(
    # Question Answering (English)
    "2wikimqa"
    "hotpotqa"
    "multifieldqa_en"
    "musique"
    "narrativeqa"
    "qasper"
    "triviaqa"
    
    # Summarization (English)
    "gov_report"
    "multi_news"
    "qmsum"
    "samsum"
    "vcsum"
    
    # Classification & Retrieval (English)
    "passage_count"
    "passage_retrieval_en"
    "trec"
    
    # Chinese tasks
    "dureader"
    "lcc"
    "lsht"
    "multifieldqa_zh"
    "passage_retrieval_zh"
    
    # Code/Repository
    "repobench-p"
    
    # Extended versions (_e suffix)
    "2wikimqa_e"
    "gov_report_e"
    "hotpotqa_e"
    "lcc_e"
    "multi_news_e"
    "multifieldqa_en_e"
    "passage_count_e"
    "passage_retrieval_en_e"
    "qasper_e"
    "repobench-p_e"
    "samsum_e"
    "trec_e"
    "triviaqa_e"
)

# 7 ScorerPress methods (excluding snapkv and observed_attention)
METHODS=(
    "tova"
    # "expected_attention"
    "streaming_llm"
    "knorm"
    "keydiff"
    "qfilter"
    # "compactor"
)

# 4 compression ratios
COMPRESSION_RATIOS=(0.1 0.2 0.3 0.5)

# GPU assignments (0-6)
GPUS=( 1 2  4 5 6)

# Output directory
OUTPUT_DIR="./longbenchresult"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/qwen3_longbench_parallel_${TIMESTAMP}.log"

echo "============================================" | tee -a "$MAIN_LOG"
echo "Qwen3-8B LongBench Parallel Evaluation" | tee -a "$MAIN_LOG"
echo "============================================" | tee -a "$MAIN_LOG"
echo "Model: $MODEL" | tee -a "$MAIN_LOG"
echo "Dataset: $DATASET" | tee -a "$MAIN_LOG"
echo "LongBench Tasks: ${LONGBENCH_TASKS[*]}" | tee -a "$MAIN_LOG"
echo "Methods: ${METHODS[*]}" | tee -a "$MAIN_LOG"
echo "Compression Ratios: ${COMPRESSION_RATIOS[*]}" | tee -a "$MAIN_LOG"
echo "GPUs: ${GPUS[*]}" | tee -a "$MAIN_LOG"
echo "Output Directory: $OUTPUT_DIR" | tee -a "$MAIN_LOG"
echo "Start Time: $(date)" | tee -a "$MAIN_LOG"
echo "============================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Function to run evaluation for one method on one GPU with all compression ratios and tasks
run_method() {
    local method=$1
    local gpu=$2
    local method_log="$LOG_DIR/qwen3_longbench_${method}_gpu${gpu}_${TIMESTAMP}.log"
    
    echo "[GPU $gpu] Starting evaluation for method: $method" | tee -a "$MAIN_LOG"
    echo "[GPU $gpu] Log file: $method_log" | tee -a "$MAIN_LOG"
    
    {
        echo "=========================================="
        echo "Method: $method"
        echo "GPU: $gpu"
        echo "Start Time: $(date)"
        echo "=========================================="
        echo ""
        
        for task in "${LONGBENCH_TASKS[@]}"; do
            echo "========================================"
            echo "LongBench Task: $task"
            echo "========================================"
            
            for ratio in "${COMPRESSION_RATIOS[@]}"; do
                echo "----------------------------------------"
                echo "Task: $task | Compression Ratio: $ratio"
                echo "----------------------------------------"
                
                CUDA_VISIBLE_DEVICES=$gpu python evaluate.py \
                    --dataset "$DATASET" \
                    --data_dir "$task" \
                    --model "$MODEL" \
                    --press_name "$method" \
                    --compression_ratio "$ratio" \
                    --device cuda:0 \
                    --fraction 1.0 \
                    --output_dir "$OUTPUT_DIR" \
                    2>&1
                
                local exit_code=$?
                if [ $exit_code -eq 0 ]; then
                    echo "✓ Completed: $method (task=$task, ratio=$ratio) on GPU $gpu" | tee -a "$MAIN_LOG"
                else
                    echo "✗ Failed: $method (task=$task, ratio=$ratio) on GPU $gpu (exit code: $exit_code)" | tee -a "$MAIN_LOG"
                fi
                echo ""
            done
            echo ""
        done
        
        echo "=========================================="
        echo "Method: $method completed"
        echo "End Time: $(date)"
        echo "=========================================="
        
    } > "$method_log" 2>&1 &
    
    # Store the PID for this background job
    eval "PID_${method}=$!"
    echo "[GPU $gpu] Started background job for $method (PID: ${!})" | tee -a "$MAIN_LOG"
}

# Launch all methods in parallel, one per GPU
echo "Launching parallel evaluations..." | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

for i in "${!METHODS[@]}"; do
    method="${METHODS[$i]}"
    gpu="${GPUS[$i]}"
    run_method "$method" "$gpu"
    sleep 2  # Small delay to avoid simultaneous model loading
done

echo "" | tee -a "$MAIN_LOG"
echo "All jobs launched. Waiting for completion..." | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Wait for all background jobs to complete
wait

echo "" | tee -a "$MAIN_LOG"
echo "============================================" | tee -a "$MAIN_LOG"
echo "All evaluations completed!" | tee -a "$MAIN_LOG"
echo "End Time: $(date)" | tee -a "$MAIN_LOG"
echo "============================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Summary
echo "Summary of Results:" | tee -a "$MAIN_LOG"
echo "-------------------" | tee -a "$MAIN_LOG"

for method in "${METHODS[@]}"; do
    echo "" | tee -a "$MAIN_LOG"
    echo "Method: $method" | tee -a "$MAIN_LOG"
    for task in "${LONGBENCH_TASKS[@]}"; do
        echo "  Task: $task" | tee -a "$MAIN_LOG"
        for ratio in "${COMPRESSION_RATIOS[@]}"; do
            # Find result directory
            result_pattern="${OUTPUT_DIR}/longbench__${task}__${MODEL}__${method}__${ratio}*"
            result_dirs=($(ls -d $result_pattern 2>/dev/null || echo ""))
            
            if [ ${#result_dirs[@]} -gt 0 ]; then
                latest_dir="${result_dirs[-1]}"
                if [ -f "$latest_dir/metrics.json" ]; then
                    echo "    Ratio $ratio: ✓ (Results: $latest_dir)" | tee -a "$MAIN_LOG"
                else
                    echo "    Ratio $ratio: ⚠ (Directory exists but no metrics: $latest_dir)" | tee -a "$MAIN_LOG"
                fi
            else
                echo "    Ratio $ratio: ✗ (No results found)" | tee -a "$MAIN_LOG"
            fi
        done
    done
done

echo "" | tee -a "$MAIN_LOG"
echo "Main log: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "Individual logs: $LOG_DIR/qwen3_longbench_*_${TIMESTAMP}.log" | tee -a "$MAIN_LOG"
echo "Results directory: $OUTPUT_DIR" | tee -a "$MAIN_LOG"

