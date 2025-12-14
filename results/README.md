# Experimental Results

This directory contains logs and analysis from all experimental runs.

## Directory Structure

```
results/
├── 1gpu_baseline/          # Single GPU baseline experiment
│   ├── 1gpu_9061890.out    # Training log (SUCCESS)
│   ├── 1gpu_9061890.err    # Error log (minimal warnings)
│   └── analysis.txt        # Performance metrics
│
├── 4gpu_attempt1/          # First 4 GPU attempt (config error)
│   ├── 4gpu_9097883.out    # Training log
│   ├── 4gpu_9097883.err    # Error log
│   └── analysis.txt        # Issue diagnosis
│
├── 4gpu_attempt2/          # Second 4 GPU attempt (DDP crash)
│   ├── 4gpu_9104940.out    # Training log (crashed)
│   ├── 4gpu_9104940.err    # Error log (DDP error)
│   └── analysis.txt        # Root cause analysis
│
└── 4gpu_highmem/           # High-memory extended training
    ├── 4gpu_9097894.out    # Training log (30k iterations)
    ├── 4gpu_9097894.err    # Error log
    └── analysis.txt        # Convergence analysis
```

## Key Results Summary

### 1 GPU Baseline (SUCCESS ✓)
```
Job ID: 9061890
Duration: 679 seconds (11.3 minutes)
Iterations: 7,000
Average time/iter: 82 ms
Throughput: 12.5 M rays/second
GPU Utilization: 95%+
Memory Usage: 28 GB / 40 GB
Final Gaussians: 54,275 primitives
Status: COMPLETED SUCCESSFULLY
```

**Performance Breakdown:**
- Rasterization: 45ms (55%)
- Backward pass: 30ms (37%)
- Optimizer: 7ms (8%)

**Key Log Snippet:**
```
Step (% Done)       Train Iter (time)    ETA (time)           Train Rays / Sec
-----------------------------------------------------------------------------------
6900 (98.57%)       82.809 ms            8 s, 280.909 ms      12.47 M
6910 (98.71%)       85.242 ms            7 s, 671.811 ms      12.10 M
6920 (98.86%)       82.742 ms            6 s, 619.342 ms      12.28 M
```

### 4 GPU Attempt 1 (Configuration Error ⚠️)
```
Job ID: 9097883
Duration: 617 seconds (10.3 minutes)
Iterations: 7,000
Throughput: 12.3 M rays/second (SAME as 1 GPU!)
GPUs Allocated: 4
GPUs Used: 1 (ERROR!)
Config Issue: num_devices=1
Speedup: 1.10× (just variance, not real)
Status: COMPLETED but inefficient
```

**Issue Identified:**
```
MachineConfig(seed=42, num_devices=1, ...)  # ❌ Should be 4!
```

**Evidence from Log:**
- Throughput identical to single GPU (12.3 vs 12.5 M rays/sec)
- Training time similar to baseline (617 vs 679 seconds)
- Only GPU 0 showed high utilization

**Root Cause:**
SLURM allocated 4 GPUs but Nerfstudio defaulted to num_devices=1 because `--machine.num-devices 4` flag was not specified.

### 4 GPU Attempt 2 (DDP Synchronization Error ❌)
```
Job ID: 9104940
Duration: 79 seconds (CRASH during initialization)
Config: num_devices=4 ✓ (CORRECTED)
GPUs Initialized: 4 ✓
Error: DDP parameter stride mismatch
Status: CRASHED
```

**Error Message:**
```
RuntimeError: params[1] in this process with sizes [54275, 3] 
appears not to match strides of the same param in process 0.
```

**Technical Details:**
- All 4 processes spawned successfully
- COLMAP point cloud loaded independently on each GPU
- Identical data but different memory layout ordering
- DDP parameter verification failed

**From Error Log (Line 120):**
```
File "torch/nn/parallel/distributed.py", line 835, in __init__
  _verify_param_shape_across_processes(self.process_group, parameters)
File "torch/distributed/utils.py", line 282
  return dist._verify_params_across_processes(...)
RuntimeError: params[1] in this process with sizes [54275, 3]
            appears not to match strides of the same param in process 0.
```

**Root Cause:**
Each GPU process independently loaded `points3D.bin` from COLMAP. File I/O timing introduced different loading orders across processes, creating identical point values but different memory layouts (strides).

### 4 GPU High-Memory (Extended Training)
```
Job ID: 9097894
Duration: 3,693 seconds (61.6 minutes)
Iterations: 30,000 (not 7,000!)
Throughput: 8.0 M rays/second
Normalized to 7k: ~862 seconds (14.4 min)
Status: COMPLETED (but also had num_devices=1 issue)
```

**Purpose:** Evaluate convergence at higher iteration counts

**Observations:**
- Lower throughput than baseline (8.0 vs 12.5 M rays/sec)
- Suggests system load or configuration differences
- Also only used 1 GPU despite allocating 4

## Metrics Comparison

| Configuration | Time (s) | Time (min) | Throughput | GPUs Used | Status |
|---------------|----------|------------|------------|-----------|--------|
| 1 GPU Baseline | 679 | 11.3 | 12.5 M | 1/1 ✓ | Success |
| 4 GPU Attempt 1 | 617 | 10.3 | 12.3 M | 1/4 ❌ | Config error |
| 4 GPU Attempt 2 | 79 | 1.3 | - | 4/4 ✓ (crashed) | DDP error |
| 4 GPU High-Mem | 3,693* | 61.6* | 8.0 M | 1/4 ❌ | Config error |
| **Expected 4 GPU** | **~226** | **~3.8** | **~50 M** | **4/4** | **Theoretical** |

*30,000 iterations, normalized to 7k: ~862s

## Expected vs Actual Performance

**Theoretical Analysis (if DDP worked):**
- Expected speedup: 2.8-3.2× (70-80% efficiency)
- Expected time: 180-240 seconds
- Expected throughput: 40-50 M rays/sec

**Actual Challenge 1 (Config Error):**
- Actual speedup: 1.10× (measurement variance)
- Wasted resources: 3 GPUs allocated but idle
- Lesson: Multi-layer configuration validation needed

**Actual Challenge 2 (DDP Sync):**
- Crash during initialization
- Challenge: Data-dependent init vs DDP assumptions
- Solution: Deterministic initialization required

## Log File Locations

Original logs from Nova cluster:
- `/work/mech-ai-scratch/mohd7/dist-3dgs/logs/1gpu_9061890.out`
- `/work/mech-ai-scratch/mohd7/dist-3dgs/logs/4gpu_9097883.out`
- `/work/mech-ai-scratch/mohd7/dist-3dgs/logs/4gpu_9104940.out`
- `/work/mech-ai-scratch/mohd7/dist-3dgs/logs/4gpu_9097894.out`

## Key Findings

### Challenge 1: Configuration Layer Mismatch
**Symptom:** Multi-GPU allocation but single-GPU usage  
**Cause:** `num_devices=1` despite `--gres=gpu:4`  
**Solution:** Explicit `--machine.num-devices N` flag  
**Impact:** Wasted HPC resources (3 idle GPUs)

### Challenge 2: DDP Parameter Synchronization
**Symptom:** RuntimeError during DDP initialization  
**Cause:** Non-deterministic COLMAP loading across processes  
**Solution:** Rank 0 loads + sorts + broadcasts  
**Impact:** Training cannot proceed to completion

## Reproducibility

All results can be reproduced by:
1. Using the exact SLURM scripts in `slurm_scripts/`
2. Same dataset: MipNeRF360 bicycle scene
3. Same hardware: NVIDIA A100 GPUs
4. Same software: Nerfstudio 1.0.2, gsplat 0.1.11, PyTorch 2.0.1

Expected variance: ±5% due to system load and scheduling

---

**Last Updated:** December 2024  
**Author:** Mohammed Musthafa Rafi
