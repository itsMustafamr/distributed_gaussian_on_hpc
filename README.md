# Distributed 3D Gaussian Splatting Training on HPC

**Author:** Mohammed Musthafa Rafi  
**Course:** COMS 625
**Institution:** Iowa State University  
**Date:** December 14 2025

## Overview

This project investigates distributed training strategies for 3D Gaussian Splatting (3DGS) using PyTorch Distributed Data Parallsel on Iowa State's Nova HPC cluster. Through iterative experimentation, we identified two critical technical challenges in distributed neural rendering.

## Research Questions

1. Can 3D Gaussian Splatting training be accelerated using multi-GPU distributed training?
2. What technical challenges arise when applying standard distributed training frameworks to dynamic neural rendering architectures?
3. How do configuration parameters across HPC system layers (SLURM, framework, runtime) impact distributed training success?

## Key Findings

### Challenge 1: Configuration Layer Mismatch
- **Issue:** SLURM allocated 4 GPUs but framework used only 1 due to `num_devices=1` parameter
- **Impact:** No speedup despite resource allocation
- **Solution:** Explicit `--machine.num-devices` flag required

### Challenge 2: DDP Parameter Synchronization
- **Issue:** Data-dependent initialization from COLMAP creates non-deterministic memory layouts
- **Error:** `RuntimeError: params[1] appears not to match strides`
- **Root Cause:** Independent point cloud loading across GPU processes
- **Solution:** Deterministic initialization with rank 0 broadcasting

## Experimental Results

| Configuration | Time (s) | Speedup | Status | Issue |
|--------------|----------|---------|--------|-------|
| 1 GPU Baseline | 679 | 1.0× | ✓ Success | None |
| 4 GPU Attempt 1 | 617 | 1.10× | ⚠ Config Error | `num_devices=1` |
| 4 GPU Attempt 2 | 79 | - | ❌ Crash | DDP stride mismatch |
| 4 GPU Expected | ~226 | ~3.0× | 📊 Theoretical | 75% efficiency |

## Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── slurm_scripts/                     # SLURM job submission scripts
│   ├── train_1gpu.slurm              # Single GPU baseline
│   ├── train_4gpu_attempt1.slurm     # First 4 GPU attempt (config error)
│   ├── train_4gpu_attempt2.slurm     # Second 4 GPU attempt (DDP error)
│   ├── train_4gpu_highmem.slurm      # High-memory extended training
│   └── README.md                      # SLURM scripts documentation
│
├── scripts/                           # Helper scripts
│   ├── setup_dataset.sh              # Download and process MipNeRF360 data
│   ├── verify_setup.py               # Verify environment and dependencies
│   └── analyze_logs.py               # Parse training logs and extract metrics
│
├── results/                           # Experimental results
│   ├── 1gpu_baseline/                # Single GPU results
│   ├── 4gpu_attempt1/                # First 4 GPU attempt results
│   ├── 4gpu_attempt2/                # Second 4 GPU attempt results
│   ├── 4gpu_highmem/                 # High-memory results
│   └── README.md                      # Results documentation
│
├── figures/                           # Generated plots and visualizations
│   ├── generate_plots.py             # Script to create all figures
│   └── README.md                      # Figure descriptions
│
└── docs/                              # Documentation
    ├── SETUP.md                       # Detailed setup instructions
    ├── TROUBLESHOOTING.md            # Common issues and solutions
    └── CONTRIBUTIONS.md              # My specific contributions
```

## Quick Start

### Prerequisites

- Access to HPC cluster with NVIDIA GPUs (tested on A100)
- CUDA 11.8+
- Conda/Miniconda
- SLURM job scheduler

### Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/distributed-3dgs.git
cd distributed-3dgs

# 2. Create conda environment
conda env create -f environment.yml
conda activate dist3dgs

# 3. Install Nerfstudio and dependencies
pip install nerfstudio
pip install gsplat

# 4. Download and process dataset
bash scripts/setup_dataset.sh

# 5. Verify setup
python scripts/verify_setup.py
```

### Running Experiments

#### Single GPU Baseline
```bash
sbatch slurm_scripts/train_1gpu.slurm
```

#### Multi-GPU Training (Corrected Configuration)
```bash
sbatch slurm_scripts/train_4gpu_attempt2.slurm
```

### Monitoring Training

```bash
# Check job status
squeue -u $USER

# Monitor training progress
tail -f logs/1gpu_*.out

# Check GPU usage (on compute node)
ssh <node-name>
nvidia-smi
```

## Reproducibility

### Hardware Environment
- **GPUs:** NVIDIA A100 (40GB HBM2e)
- **CPUs:** AMD EPYC 7763
- **Memory:** 512 GB DDR4 per node
- **Interconnect:** InfiniBand HDR (200 Gbps)
- **Cluster:** Iowa State University Nova HPC

### Software Environment
- **OS:** Ubuntu 22.04 LTS
- **CUDA:** 11.8
- **Python:** 3.10
- **PyTorch:** 2.0.1
- **Nerfstudio:** 1.0.2
- **gsplat:** 0.1.11

### Dataset
- **Name:** MipNeRF360 Bicycle Scene
- **Source:** http://storage.googleapis.com/gresearch/refraw360/bicycle.zip
- **Images:** 188 views at 1920×1080 (downscaled 4× to 480×270)
- **COLMAP:** Pre-computed sparse reconstruction with 54,275 points

### Training Configuration
- **Iterations:** 7,000
- **Optimizer:** Adam
- **Learning Rates:**
  - Position: 1.6e-4 → 1.6e-6 (exponential decay)
  - Spherical Harmonics: 2.5e-3 (DC), 1.25e-4 (rest)
  - Opacity: 0.05
  - Scale: 0.005
  - Rotation: 0.001
- **Batch Size:** 4,096 rays per GPU
- **Densification:** Every 100 iterations until 15,000

## Results Analysis

### Performance Metrics

**Single GPU Baseline:**
- Training time: 679 seconds (11.3 minutes)
- Throughput: 12.5 million rays/second
- GPU utilization: 95%+
- Memory usage: 28 GB / 40 GB

**4 GPU Attempt 1 (Configuration Error):**
- Training time: 617 seconds (10.3 minutes)
- Throughput: 12.3 million rays/second (same as 1 GPU!)
- Issue: Only used 1 GPU despite allocating 4
- Root cause: `num_devices=1` in configuration

**4 GPU Attempt 2 (DDP Synchronization Error):**
- Crash time: 79 seconds (during initialization)
- Issue: DDP parameter stride mismatch
- Root cause: Non-deterministic COLMAP loading

### Generate Analysis Plots

```bash
cd figures/
python generate_plots.py
```

This creates:
- `experimental_journey.png` - Timeline of all attempts
- `technical_challenges.png` - Detailed error analysis
- `performance_comparison.png` - Expected vs actual results

## My Contributions

This project represents my independent research work under the supervision of Dr. Adarsh Krishnamurthy and Dr. Aditya Balu. Specific contributions include:

1. **System Design:**
   - Designed SLURM job scripts for single and multi-GPU configurations
   - Configured Nerfstudio training pipeline for HPC environment
   - Set up dataset preprocessing and validation workflows

2. **Experimental Investigation:**
   - Conducted baseline single-GPU training experiments
   - Performed iterative multi-GPU training attempts
   - Identified and diagnosed configuration layer mismatch
   - Discovered DDP synchronization incompatibility

3. **Technical Analysis:**
   - Root cause analysis of configuration parameter misalignment
   - Deep dive into PyTorch DDP parameter verification process
   - Analysis of data-dependent initialization challenges
   - Comparison with traditional neural network distributed training

4. **Documentation:**
   - Comprehensive documentation of all experiments
   - Error message analysis and interpretation
   - Proposed solutions for both technical challenges
   - Reproducibility instructions and environment setup

5. **Software Integration:**
   - Integration of Nerfstudio with SLURM job scheduler
   - Dataset preparation pipeline for MipNeRF360
   - Log parsing and metrics extraction scripts
   - Visualization tools for experimental results

### External Dependencies

- **Nerfstudio:** Framework for training neural radiance fields (Tancik et al., 2023)
- **gsplat:** CUDA kernels for Gaussian splatting rasterization
- **PyTorch DDP:** Distributed data parallel training framework
- **COLMAP:** Structure-from-Motion preprocessing (Schönberger & Frahm, 2016)
- **MipNeRF360 Dataset:** Google Research benchmark dataset

All external tools and datasets are properly cited in the project report.

## Known Issues

### Issue 1: Configuration Parameter Mismatch
**Status:** Identified and documented  
**Impact:** Single GPU usage despite multi-GPU allocation  
**Solution:** Use `--machine.num-devices N` flag explicitly

### Issue 2: DDP Parameter Stride Mismatch
**Status:** Identified, solution proposed but not yet implemented  
**Impact:** Training crashes during DDP initialization  
**Solution:** Implement deterministic point cloud initialization (see `docs/TROUBLESHOOTING.md`)

## Future Work

1. **Implement Deterministic Initialization:**
   - Modify Nerfstudio COLMAP loading to ensure consistent ordering
   - Rank 0 loads and broadcasts point cloud to all processes
   - Verify parameter stride consistency before DDP initialization

2. **Test Proposed Solutions:**
   - Run corrected 4 GPU training with deterministic initialization
   - Measure actual distributed training performance
   - Compare against theoretical 75% efficiency prediction

3. **Scale Beyond 4 GPUs:**
   - Test 8 GPU configuration
   - Investigate multi-node distributed training
   - Explore gradient compression for reduced communication

4. **Alternative Parallelization Strategies:**
   - Spatial partitioning for model parallelism
   - Hybrid data/model parallelism for very large scenes
   - Asynchronous parameter updates

## Citation

If you use this work, please cite:

```bibtex
@misc{rafi2024distributed3dgs,
  author = {Rafi, Mohammed Musthafa},
  title = {Distributed 3D Gaussian Splatting Training on HPC},
  year = {2024},
  institution = {Iowa State University},
  howpublished = {\url{https://github.com/yourusername/distributed-3dgs}}
}
```

## References

1. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM TOG (SIGGRAPH).

2. Tancik, M., Weber, E., Ng, E., et al. (2023). Nerfstudio: A Modular Framework for Neural Radiance Field Development. arXiv:2302.04264.

3. Li, S., Zhao, Y., Varma, R., et al. (2020). PyTorch Distributed: Experiences on Accelerating Data Parallel Training. VLDB.

4. Schönberger, J. L., & Frahm, J. M. (2016). Structure-from-Motion Revisited. CVPR.

## License

This project is for educational and research purposes. Please respect the licenses of all dependencies.

## Contact

**Mohammed Musthafa Rafi**  
Iowa State University  
Email: mohd7@iastate.edu

For questions or collaboration opportunities, please open an issue on GitHub or contact via email.

---

**Acknowledgments:** This research was conducted on Iowa State University's Nova HPC cluster. Thanks to Research IT for technical support and cluster access. Special thanks to advisors Dr. Adarsh Krishnamurthy and Dr. Aditya Balu for guidance throughout this project.
