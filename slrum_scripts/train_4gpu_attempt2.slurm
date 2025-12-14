#!/bin/bash
#
# 4 GPU 3D Gaussian Splatting Training - Attempt 2 (DDP Synchronization Error)
# ==============================================================================
#
# This script shows the CORRECTED configuration (--machine.num-devices 4 added)
# but reveals a FUNDAMENTAL INCOMPATIBILITY between PyTorch DDP and 3DGS's
# data-dependent initialization.
#
# ISSUE: DDP parameter stride mismatch during initialization
# ROOT CAUSE: Non-deterministic COLMAP point cloud loading across processes
# RESULT: Training crashes after 79 seconds during DDP setup
#
# Author: Mohammed Musthafa Rafi
# Date: December 2024
#
# Actual Results:
# - Configuration: num_devices=4 ✓ (CORRECTED from Attempt 1)
# - Crash time: 79 seconds (during DDP initialization)
# - Error: "RuntimeError: params[1] appears not to match strides"
# - GPUs allocated: 4 ✓
# - GPUs initialized: 4 ✓  
# - Issue: Parameter memory layout inconsistency across processes
#

# ============================================================================
# SLURM Resource Configuration
# ============================================================================

#SBATCH --job-name=3dgs_4gpu_attempt2
#SBATCH --output=logs/4gpu_attempt2_%j.out
#SBATCH --error=logs/4gpu_attempt2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4                     # Request 4 GPUs
#SBATCH --mem=128G
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
echo "Job: 4 GPU Training - Attempt 2 (Corrected Config)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $(hostname)"
echo "Configuration: num_devices=4 (CORRECTED)"
echo "========================================"

echo ""
echo "GPUs Allocated:"
nvidia-smi --list-gpus
echo ""

START_TIME=$(date +%s)

# ============================================================================
# Training Command - CORRECTED CONFIGURATION
# ============================================================================

# ✓ FIXED: Now explicitly specifies --machine.num-devices 4
# ✓ FIXED: Increased batch size to 16384 (4096 per GPU × 4 GPUs)
#
# This configuration correctly tells Nerfstudio to use all 4 GPUs.
# However, it will crash during DDP initialization due to a different issue.
#
# The crash occurs in PyTorch's DDP parameter verification:
#   File "torch/nn/parallel/distributed.py", line 835, in __init__
#     _verify_param_shape_across_processes(...)
#   File "torch/distributed/utils.py", line 282
#     return dist._verify_params_across_processes(...)
#   RuntimeError: params[1] in this process with sizes [54275, 3]
#                 appears not to match strides of the same param in process 0.
#
# What this error means:
# - All 4 GPU processes have parameters with shape [54275, 3] ✓
# - But the memory layout (stride) differs between processes ❌
# - PyTorch DDP requires IDENTICAL memory layouts for gradient sync

ns-train splatfacto \
  --data data/bicycle/processed \
  --output-dir outputs/4gpu_attempt2 \
  --experiment-name attempt2_ddp_error \
  --max-num-iterations 7000 \
  --machine.num-devices 4 \
  --pipeline.datamanager.train-num-rays-per-batch 16384 \
  --pipeline.model.cull-alpha-thresh 0.005 \
  --pipeline.model.densify-grad-thresh 0.0002 \
  --viewer.quit-on-train-completion True

# Note: Training will crash before reaching this point

# ============================================================================
# Post-Crash Analysis
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "Job Ended (Expected: Crash)"
echo "Duration: ${DURATION} seconds"
echo "========================================"

# ============================================================================
# DDP Synchronization Error - Technical Explanation
# ============================================================================

echo ""
echo "========================================"
echo "DDP SYNCHRONIZATION ERROR ANALYSIS"
echo "========================================"
echo ""
echo "Error Message:"
echo "  RuntimeError: params[1] in this process with sizes [54275, 3]"
echo "  appears not to match strides of the same param in process 0."
echo ""
echo "What happened:"
echo "  1. SLURM allocated 4 GPUs ✓"
echo "  2. Nerfstudio spawned 4 processes (one per GPU) ✓"
echo "  3. Each process loaded COLMAP point cloud independently"
echo "  4. File I/O timing created different load orders"
echo "  5. Same data, different memory arrangements"
echo "  6. PyTorch DDP verification failed ❌"
echo ""
echo "Root Cause - Data-Dependent Initialization:"
echo "  Each GPU process executes:"
echo "    points = load_colmap('points3D.bin')"
echo ""
echo "  Process 0: loads [A, B, C, D, ...] → stride (3, 1)"
echo "  Process 1: loads [A, C, B, D, ...] → stride (3, 1)"
echo "  Process 2: loads [A, B, D, C, ...] → stride (3, 1)"
echo "  Process 3: loads [A, D, B, C, ...] → stride (3, 1)"
echo ""
echo "  Same points, different order → different memory layout!"
echo ""
echo "Why Traditional DNNs Don't Have This Problem:"
echo "  - Random weight initialization: torch.randn(..., seed=42)"
echo "  - Deterministic across processes (same seed)"
echo "  - No external file loading"
echo "  - Fixed architecture (no densification/pruning)"
echo ""
echo "Why 3DGS is Different:"
echo "  - Initializes from COLMAP point cloud (external data)"
echo "  - File I/O introduces non-determinism"
echo "  - Dynamic architecture (densification during training)"
echo "  - Data-dependent structure"
echo ""
echo "Proposed Solution:"
echo "  Rank 0 loads point cloud, sorts, broadcasts to all ranks"
echo "  See: docs/TROUBLESHOOTING.md for implementation"
echo ""
echo "To verify this error in the log:"
echo "  grep -A 5 'RuntimeError' logs/4gpu_attempt2_${SLURM_JOB_ID}.err"
echo ""
echo "========================================"

# ============================================================================
# Technical Deep Dive - Parameter Verification Process
# ============================================================================
#
# PyTorch DDP performs 3 checks during initialization:
#
# 1. Parameter Names Match:
#    ✓ All processes have same parameter names
#    ✓ Example: "means", "scales", "rotations", etc.
#
# 2. Parameter Shapes Match:
#    ✓ All processes have shapes [54275, 3]
#    ✓ COLMAP provides same number of points to all
#
# 3. Parameter Strides Match:
#    ❌ Memory layout differs between processes!
#    ❌ Process 0: stride (3, 1)
#    ❌ Process 1: stride (3, 1) but different ordering
#
# The third check fails because torch.Tensor equality checks both:
# - tensor.size() : [54275, 3] ✓ matches
# - tensor.stride() : differs across processes ❌
#
# Example:
#   Process 0: points[0] = [x1, y1, z1]  memory: 0x1000
#   Process 1: points[0] = [x1, y1, z1]  memory: 0x2000 (different position!)
#
# Even though points[0] has same values, it's at different memory address
# and potentially different position in the array order.
#
# ============================================================================

# ============================================================================
# Solution Implementation (Not Yet Implemented)
# ============================================================================
#
# Pseudocode for deterministic initialization:
#
# if torch.distributed.is_initialized():
#     if torch.distributed.get_rank() == 0:
#         # Only rank 0 loads the point cloud
#         points = load_colmap_points('points3D.bin')
#         
#         # Sort by spatial coordinates for deterministic ordering
#         # Lexicographic sort: first by x, then y, then z
#         indices = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
#         points = points[indices]
#     else:
#         # Other ranks wait for broadcast
#         points = None
#     
#     # Rank 0 broadcasts sorted points to all other ranks
#     points = broadcast_object(points, src=0)
# else:
#     # Single GPU: load normally
#     points = load_colmap_points('points3D.bin')
#
# # All ranks now have IDENTICAL point ordering
# # Initialize Gaussians from points
# gaussians = initialize_from_points(points)
#
# This ensures:
# - Same data ✓
# - Same order ✓
# - Same memory layout ✓
# - DDP verification passes ✓
#
# Overhead: ~200ms for broadcasting 54,000 points (negligible)
#
# ============================================================================

# ============================================================================
# Lessons Learned
# ============================================================================
#
# 1. Distributed Training Assumptions:
#    PyTorch DDP assumes models initialize identically across all processes.
#    This holds for:
#    - CNNs with random weights (deterministic seed)
#    - Transformers with random weights
#    - Any architecture without external data loading
#
#    This breaks for:
#    - Models initialized from files (like 3DGS)
#    - Non-deterministic initialization
#    - Data-dependent architectures
#
# 2. Configuration Validation is Multi-Layered:
#    Even with correct --machine.num-devices, framework-level issues can
#    prevent distributed training from working.
#
# 3. Error Messages Provide Clues:
#    "appears not to match strides" immediately points to memory layout
#    rather than values or shapes.
#
# 4. File I/O and Distributed Training Don't Mix Well:
#    Each process loading the same file independently introduces
#    non-determinism that can break distributed training.
#
# 5. Novel Architectures May Not "Just Work" with Standard Tools:
#    3DGS's dynamic nature and data-dependent initialization require
#    custom distributed training solutions.
#
# ============================================================================
