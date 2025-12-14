#!/bin/bash
#
# 4 GPU 3D Gaussian Splatting Training - Attempt 1 (Configuration Error)
# ========================================================================
#
# This script demonstrates a COMMON CONFIGURATION ERROR in distributed
# training on HPC systems. The SLURM scheduler successfully allocates
# 4 GPUs, but the training framework only uses 1 GPU.
#
# ISSUE: Missing --machine.num-devices 4 flag
# RESULT: Only 1 GPU utilized despite 4 being allocated
# LESSON: HPC configuration requires alignment across multiple layers
#
# Author: Mohammed Musthafa Rafi
# Date: December 2024
#
# Actual Results:
# - Training time: 617 seconds (10.3 minutes)
# - Throughput: 12.3 M rays/sec (SAME as single GPU!)
# - GPUs allocated: 4
# - GPUs actually used: 1
# - Speedup: 1.10× (just measurement variance, not real acceleration)
#

# ============================================================================
# SLURM Resource Configuration
# ============================================================================

#SBATCH --job-name=3dgs_4gpu_attempt1
#SBATCH --output=logs/4gpu_attempt1_%j.out
#SBATCH --error=logs/4gpu_attempt1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                    # 8 CPUs per GPU × 4 GPUs
#SBATCH --gres=gpu:a100:4                     # ✓ Request 4 A100 GPUs (SUCCESSFUL)
#SBATCH --mem=128G                            # More memory for 4 GPUs
#SBATCH --time=01:00:00
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
echo "Job: 4 GPU Training - Attempt 1"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $(hostname)"
echo "========================================"

# Verify 4 GPUs were allocated by SLURM
echo ""
echo "GPUs Allocated by SLURM:"
nvidia-smi --list-gpus
echo ""
echo "Expected: 4 GPUs"
echo "Note: Allocation != Usage!"
echo ""

START_TIME=$(date +%s)

# ============================================================================
# Training Command - CONFIGURATION ERROR VERSION
# ============================================================================

# ❌ PROBLEM: This command does NOT specify how many GPUs to use!
#
# The SLURM scheduler allocated 4 GPUs (--gres=gpu:a100:4 above)
# BUT Nerfstudio defaults to num_devices=1 when not specified
#
# What happens:
# 1. SLURM allocates GPUs 0, 1, 2, 3 to this job
# 2. Nerfstudio sees all 4 GPUs are available
# 3. But MachineConfig defaults to num_devices=1
# 4. Training runs on GPU 0 only
# 5. GPUs 1, 2, 3 sit idle (wasted resources!)
#
# How to verify the error in the log:
#   grep "num_devices" logs/4gpu_attempt1_*.out
#   Should show: num_devices=1 ❌
#
# The correct version is in train_4gpu_attempt2.slurm

ns-train splatfacto \
  --data data/bicycle/processed \
  --output-dir outputs/4gpu_attempt1 \
  --experiment-name attempt1_config_error \
  --max-num-iterations 7000 \
  --pipeline.model.cull-alpha-thresh 0.005 \
  --pipeline.model.densify-grad-thresh 0.0002 \
  --viewer.quit-on-train-completion True
  # ❌ MISSING: --machine.num-devices 4
  # ❌ MISSING: --pipeline.datamanager.train-num-rays-per-batch 16384

# ============================================================================
# Post-Training Analysis
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))

echo ""
echo "========================================"
echo "Training Complete!"
echo "Duration: ${DURATION} seconds (${MINUTES} minutes)"
echo "========================================"

echo "Duration: ${DURATION} seconds" > outputs/4gpu_attempt1/training_time.txt
echo "Minutes: ${MINUTES}" >> outputs/4gpu_attempt1/training_time.txt

nvidia-smi > outputs/4gpu_attempt1/final_gpu_stats.txt

# ============================================================================
# Error Analysis and Verification
# ============================================================================

echo ""
echo "========================================"
echo "CONFIGURATION ERROR ANALYSIS"
echo "========================================"
echo ""
echo "Symptoms of this error:"
echo "  1. Training time similar to single GPU (~600s vs 679s)"
echo "  2. Throughput similar to single GPU (~12 M rays/sec)"
echo "  3. Only GPU 0 shows high utilization in nvidia-smi"
echo "  4. Config shows num_devices=1"
echo ""
echo "To verify:"
echo "  grep 'num_devices' logs/4gpu_attempt1_${SLURM_JOB_ID}.out"
echo "  Expected output: num_devices=1 (WRONG!)"
echo ""
echo "Root cause:"
echo "  Multi-layer configuration mismatch"
echo "  - SLURM layer: Allocated 4 GPUs ✓"
echo "  - Framework layer: Used 1 GPU ❌"
echo ""
echo "Lesson learned:"
echo "  GPU allocation ≠ GPU utilization"
echo "  Always explicitly specify --machine.num-devices"
echo "  Validate with throughput and nvidia-smi monitoring"
echo ""
echo "Corrected version available in:"
echo "  slurm_scripts/train_4gpu_attempt2.slurm"
echo ""
echo "========================================"

# ============================================================================
# Key Takeaways for Future Researchers
# ============================================================================
#
# 1. SLURM Resource Allocation (--gres=gpu:4)
#    - Reserves GPUs for exclusive use by your job
#    - Makes GPUs visible via CUDA_VISIBLE_DEVICES
#    - Does NOT tell your framework to use them!
#
# 2. Framework Configuration (--machine.num-devices)
#    - Tells Nerfstudio/PyTorch how many GPUs to use
#    - Must match SLURM allocation for efficiency
#    - Defaults to 1 if not specified!
#
# 3. Validation Methods:
#    - Check config in output log (grep "num_devices")
#    - Monitor throughput (should be ~4× higher)
#    - Use nvidia-smi on compute node (all GPUs busy)
#    - Profile GPU utilization (all ~95%+)
#
# 4. Common Indicators of This Error:
#    - Similar training time to baseline
#    - No speedup despite multi-GPU allocation
#    - Low efficiency metrics (~25%)
#    - Throughput doesn't scale with GPU count
#
# This error cost us 3 wasted GPU-hours (3 GPUs allocated but idle)
# Always validate your distributed training setup!
#
# ============================================================================
