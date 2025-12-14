#!/bin/bash
#
# 4 GPU High-Memory Extended Training (30,000 iterations)
# ========================================================
#
# This script runs extended training with increased memory allocation.
# Note: This configuration also has the num_devices=1 issue from Attempt 1,
# so it only uses 1 GPU despite allocating 4.
#
# Purpose: Evaluate convergence behavior at higher iteration counts
#
# Author: Mohammed Musthafa Rafi
# Date: December 2024
#
# Actual Results:
# - Duration: 3,693 seconds (61.6 minutes)
# - Iterations: 30,000 (vs 7,000 in other experiments)
# - Throughput: 8.0 M rays/sec (lower due to system load)
# - Issue: Same as Attempt 1 (only used 1 GPU)
#

# ============================================================================
# SLURM Resource Configuration
# ============================================================================

#SBATCH --job-name=3dgs_4gpu_highmem
#SBATCH --output=logs/4gpu_highmem_%j.out
#SBATCH --error=logs/4gpu_highmem_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G                            # Increased memory (256 GB)
#SBATCH --time=02:00:00                       # Longer time limit (2 hours)
#SBATCH --partition=gpu

# ============================================================================
# Environment Setup
# ============================================================================

module load cuda/11.8
source ~/.bashrc
conda activate dist3dgs

cd ~/dist-3dgs
mkdir -p logs

# ============================================================================
# Job Information
# ============================================================================

echo "========================================"
echo "Job: 4 GPU High-Memory Extended Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $(hostname)"
echo "Memory: 256 GB"
echo "Iterations: 30,000"
echo "========================================"

echo ""
echo "GPUs Allocated:"
nvidia-smi --list-gpus
echo ""

START_TIME=$(date +%s)

# ============================================================================
# Extended Training Configuration
# ============================================================================

# Train for 30,000 iterations to evaluate long-term convergence
# This is 4.3× longer than standard 7,000 iteration training

MAX_ITERATIONS=30000

ns-train splatfacto \
  --data data/bicycle/processed \
  --output-dir outputs/4gpu_highmem \
  --experiment-name highmem_extended \
  --max-num-iterations ${MAX_ITERATIONS} \
  --pipeline.model.cull-alpha-thresh 0.005 \
  --pipeline.model.densify-grad-thresh 0.0002 \
  --viewer.quit-on-train-completion True
  # Note: Same configuration error as Attempt 1
  # Missing: --machine.num-devices 4

# ============================================================================
# Post-Training Analysis
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))

echo ""
echo "========================================"
echo "Extended Training Complete!"
echo "Duration: ${DURATION} seconds (${MINUTES} minutes)"
echo "========================================"

# Normalize to 7,000 iterations for comparison
NORMALIZED_TIME=$((DURATION * 7000 / MAX_ITERATIONS))
NORMALIZED_MIN=$((NORMALIZED_TIME / 60))

echo ""
echo "Normalized to 7,000 iterations:"
echo "  Time: ${NORMALIZED_TIME} seconds (${NORMALIZED_MIN} minutes)"
echo ""

echo "Duration: ${DURATION} seconds" > outputs/4gpu_highmem/training_time.txt
echo "Minutes: ${MINUTES}" >> outputs/4gpu_highmem/training_time.txt
echo "Iterations: ${MAX_ITERATIONS}" >> outputs/4gpu_highmem/training_time.txt
echo "Normalized (7k): ${NORMALIZED_TIME} seconds" >> outputs/4gpu_highmem/training_time.txt

nvidia-smi > outputs/4gpu_highmem/final_gpu_stats.txt

# ============================================================================
# Convergence Analysis Notes
# ============================================================================
#
# Purpose of Extended Training:
# - Observe if model continues improving beyond 7,000 iterations
# - Check for overfitting or instability
# - Evaluate densification/pruning dynamics over longer timeframe
#
# Observations:
# - Lower throughput (8.0 vs 12.5 M rays/sec) suggests:
#   * Different system load during execution
#   * Possible memory/cache effects
#   * Or batch size/configuration differences
#
# Note: This also suffers from num_devices=1 configuration error,
# so it only used 1 GPU despite allocating 4.
#
# ============================================================================
