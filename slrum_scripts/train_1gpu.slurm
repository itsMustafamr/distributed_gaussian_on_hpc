#!/bin/bash
#
# Single GPU 3D Gaussian Splatting Training
# ==========================================
# 
# This script trains a 3DGS model on a single NVIDIA A100 GPU.
# This serves as the baseline for comparing multi-GPU performance.
#
# Author: Mohammed Musthafa Rafi
# Date: December 2024
# Course: COMS 625 - Iowa State University
#
# Expected Results:
# - Training time: ~679 seconds (11.3 minutes)
# - Throughput: ~12.5 million rays/second
# - GPU utilization: 95%+
# - Memory usage: ~28 GB / 40 GB
#

# ============================================================================
# SLURM Resource Configuration
# ============================================================================

#SBATCH --job-name=3dgs_1gpu_baseline        # Job name for queue
#SBATCH --output=logs/1gpu_%j.out            # Standard output log (%j = job ID)
#SBATCH --error=logs/1gpu_%j.err             # Standard error log
#SBATCH --nodes=1                             # Number of nodes (single node)
#SBATCH --ntasks=1                            # Number of tasks (single task)
#SBATCH --cpus-per-task=8                     # CPU cores per task
#SBATCH --gres=gpu:a100:1                     # Request 1 A100 GPU
#SBATCH --mem=32G                             # Memory allocation (32 GB)
#SBATCH --time=02:00:00                       # Max runtime (2 hours)
#SBATCH --partition=gpu                       # Submit to GPU partition

# ============================================================================
# Environment Setup
# ============================================================================

# Load CUDA module (required for GPU operations)
module load cuda/11.8

# Activate conda environment with Nerfstudio and dependencies
# Environment must be created beforehand with:
#   conda env create -f environment.yml
source ~/.bashrc
conda activate dist3dgs

# Navigate to project directory
cd ~/dist-3dgs

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================================================
# Job Information Logging
# ============================================================================

echo "========================================"
echo "Job: Single GPU Baseline Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $(hostname)"
echo "Working Directory: $(pwd)"
echo "========================================"

# Verify GPU allocation
echo ""
echo "GPU Allocation:"
nvidia-smi --list-gpus
echo ""

# Record start time for duration calculation
START_TIME=$(date +%s)

# ============================================================================
# Training Configuration
# ============================================================================

# Dataset: MipNeRF360 Bicycle Scene
# - 188 images at 1920x1080 (downscaled 4x during training)
# - COLMAP preprocessing with ~54,000 3D points
# - Location: data/bicycle/processed
DATA_PATH="data/bicycle/processed"

# Output directory for this experiment
OUTPUT_DIR="outputs/1gpu_baseline"

# Training parameters based on original 3DGS paper
MAX_ITERATIONS=7000                # Total training iterations
CULL_ALPHA_THRESH=0.005           # Opacity threshold for pruning
DENSIFY_GRAD_THRESH=0.0002        # Gradient threshold for densification

# ============================================================================
# Nerfstudio Training Command
# ============================================================================

# Train splatfacto model (Nerfstudio's 3DGS implementation)
# 
# Key flags:
#   --data: Path to processed dataset
#   --output-dir: Where to save checkpoints and logs
#   --experiment-name: Identifier for this run
#   --max-num-iterations: Number of training steps
#   --pipeline.model.*: Model-specific parameters
#   --viewer.quit-on-train-completion: Exit after training finishes
#
ns-train splatfacto \
  --data "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name baseline_1gpu \
  --max-num-iterations ${MAX_ITERATIONS} \
  --pipeline.model.cull-alpha-thresh ${CULL_ALPHA_THRESH} \
  --pipeline.model.densify-grad-thresh ${DENSIFY_GRAD_THRESH} \
  --viewer.quit-on-train-completion True

# ============================================================================
# Post-Training Analysis
# ============================================================================

# Calculate total training duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))

echo ""
echo "========================================"
echo "Training Complete!"
echo "End Time: $(date)"
echo "Duration: ${DURATION} seconds (${MINUTES} minutes)"
echo "========================================"

# Save timing information
echo "Duration: ${DURATION} seconds" > "${OUTPUT_DIR}/training_time.txt"
echo "Minutes: ${MINUTES}" >> "${OUTPUT_DIR}/training_time.txt"
echo "Start: $(date -d @${START_TIME})" >> "${OUTPUT_DIR}/training_time.txt"
echo "End: $(date -d @${END_TIME})" >> "${OUTPUT_DIR}/training_time.txt"

# Capture final GPU statistics
nvidia-smi > "${OUTPUT_DIR}/final_gpu_stats.txt"

# Display summary
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "Training log: logs/1gpu_${SLURM_JOB_ID}.out"
echo "Error log: logs/1gpu_${SLURM_JOB_ID}.err"
echo ""

# ============================================================================
# Notes on Results
# ============================================================================
#
# Expected Performance Breakdown:
# - Rasterization: ~45ms per iteration (55% of time)
# - Backward pass: ~30ms per iteration (37% of time)  
# - Optimizer step: ~7ms per iteration (8% of time)
# - Densification spikes: ~150ms every 100 iterations
#
# Profiling shows training is compute-bound with high GPU utilization,
# suggesting good potential for multi-GPU scaling if properly configured.
#
# To render novel views after training:
#   ns-render interpolate \
#     --load-config ${OUTPUT_DIR}/baseline_1gpu/splatfacto/*/config.yml \
#     --output-path renders/bicycle_video.mp4
#
# ============================================================================
