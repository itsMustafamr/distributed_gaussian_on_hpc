# My Contributions to This Project

**Author:** Mohammed Musthafa Rafi  
**Institution:** Iowa State University  
**Course:** COMS 625  
**Advisors:** Dr. Adarsh Krishnamurthy, Dr. Aditya Balu  
**Date:** December 14 2025

## Overview

This document clearly delineates my original contributions to this research project from the external tools and frameworks used. All work was conducted independently as part of this course 625

---

## My Original Contributions

### 1. **Research Design and Methodology**

**What I did:**
- Designed the experimental methodology for investigating distributed 3DGS training
- Formulated research questions about multi-GPU scaling
- Planned iterative experimental approach to identify bottlenecks

**Evidence:**
- Complete experimental timeline documented in results/
- Three distinct experimental configurations (1 GPU, 4 GPU attempt 1, 4 GPU attempt 2)
- Systematic progression from baseline to distributed configurations

---

### 2. **HPC System Configuration**

**What I did:**
- Configured Iowa State's Nova HPC cluster for 3DGS training
- Designed SLURM job submission scripts for single and multi-GPU configurations
- Implemented resource allocation strategies (CPU, memory, GPU)
- Set up conda environment for reproducible deployments

**My original scripts:**
- `slurm_scripts/train_1gpu.slurm` - Single GPU baseline (100% my work)
- `slurm_scripts/train_4gpu_attempt1.slurm` - First multi-GPU attempt (100% my work)
- `slurm_scripts/train_4gpu_attempt2.slurm` - Corrected configuration (100% my work)
- `slurm_scripts/train_4gpu_highmem.slurm` - Extended training (100% my work)

**Evidence:**
- SLURM scripts with detailed comments explaining my configuration choices
- Job logs showing successful execution on Nova cluster
- Resource utilization analysis (GPU hours, memory usage)

---

### 3. **Problem Identification and Diagnosis**

**What I did:**
- Identified configuration layer mismatch between SLURM and Nerfstudio
- Diagnosed DDP parameter stride mismatch error through log analysis
- Performed root cause analysis for both technical challenges
- Compared 3DGS behavior with traditional neural network distributed training

**Evidence:**
- Detailed error analysis in `slurm_scripts/train_4gpu_attempt2.slurm` comments
- Comparison tables showing throughput differences
- Documentation of error messages and their interpretation

**Key findings (100% my analysis):**
1. **Challenge 1:** Multi-layer configuration validation required in HPC systems
2. **Challenge 2:** Data-dependent initialization incompatible with PyTorch DDP
3. Root cause of stride mismatch: non-deterministic COLMAP loading

---

### 4. **Experimental Execution**

**What I did:**
- Executed all training experiments on Nova cluster
- Monitored training progress and collected metrics
- Extracted performance data from logs
- Analyzed throughput, GPU utilization, and memory usage

**My experimental data:**
- 1 GPU baseline: 679 seconds, 12.5 M rays/sec
- 4 GPU attempt 1: 617 seconds, configuration error identified
- 4 GPU attempt 2: 79 seconds, DDP crash documented
- High-memory run: 3,693 seconds for 30,000 iterations

**Evidence:**
- Complete training logs in `results/` directories
- Timing measurements and GPU statistics
- Performance comparison tables

---

### 5. **Solution Proposals**

**What I did:**
- Designed deterministic initialization approach for fixing DDP issue
- Proposed rank 0 broadcast pattern for consistent memory layouts
- Developed corrective configuration strategies
- Outlined future work and alternative approaches

**My proposals (100% my work):**
1. Deterministic COLMAP loading with sorting and broadcasting
2. Configuration validation checklist for HPC distributed training
3. Multi-signal verification (throughput, nvidia-smi, config inspection)

**Evidence:**
- Pseudocode for deterministic initialization in documentation
- Detailed solution explanations in SLURM script comments
- Future work section in README.md

---

### 6. **Documentation and Reproducibility**

**What I did:**
- Wrote comprehensive README with setup instructions
- Created detailed comments in all SLURM scripts
- Documented all configuration parameters and their effects
- Wrote verification and analysis scripts
- Prepared visualization scripts for results

**My original documentation:**
- `README.md` - Complete project documentation (100% my work)
- `docs/CONTRIBUTIONS.md` - This document (100% my work)
- All SLURM script comments - Detailed explanations (100% my work)
- `scripts/verify_setup.py` - Environment verification (100% my work)
- `scripts/setup_dataset.sh` - Dataset preparation (100% my work)

---

### 7. **Data Analysis and Visualization**

**What I did:**
- Analyzed training logs to extract performance metrics
- Created comparison plots showing experimental timeline
- Generated technical challenge diagrams
- Produced performance comparison visualizations

**My visualizations:**
- `experimental_journey.png` - Timeline of all attempts
- `technical_challenges_explained.png` - Error analysis diagrams
- Performance comparison tables
- Throughput analysis charts

---

## External Tools and Frameworks Used

To maintain academic integrity, I clearly identify all external dependencies:

### Frameworks and Libraries

1. **Nerfstudio** (Tancik et al., 2023)
   - Framework for training neural radiance fields
   - Provides `ns-train` and `ns-render` commands
   - Implements splatfacto model (3DGS wrapper)
   - **Citation:** Tancik et al., "Nerfstudio: A Modular Framework for Neural Radiance Field Development," arXiv:2302.04264, 2023

2. **gsplat** (Ye et al., 2024)
   - CUDA kernels for Gaussian splatting rasterization
   - Provides optimized GPU operations
   - **Citation:** Available on GitHub

3. **PyTorch** (Paszke et al., 2019)
   - Deep learning framework
   - Provides DistributedDataParallel (DDP)
   - **Citation:** Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS, 2019

4. **COLMAP** (Schönberger & Frahm, 2016)
   - Structure-from-Motion preprocessing
   - Provides sparse 3D reconstruction
   - **Citation:** Schönberger & Frahm, "Structure-from-Motion Revisited," CVPR, 2016

### Dataset

**MipNeRF360 Bicycle Scene**
- Source: Google Research
- Pre-computed COLMAP reconstruction
- 188 images, ~54,000 3D points
- **Citation:** Barron et al., "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields," CVPR, 2022

### HPC Infrastructure

**Iowa State University Nova Cluster**
- Provided by Research IT
- NVIDIA A100 GPUs
- SLURM job scheduler
- No modifications made to cluster infrastructure

---

## Contribution Breakdown by Component

| Component | My Work | External Tools |
|-----------|---------|----------------|
| **Research Design** | 100% | - |
| **SLURM Scripts** | 100% | SLURM scheduler |
| **Environment Setup** | 100% | Conda, pip |
| **Training Execution** | 100% | Nerfstudio, gsplat, PyTorch |
| **Error Diagnosis** | 100% | - |
| **Solution Proposals** | 100% | - |
| **Documentation** | 100% | - |
| **Visualizations** | 100% | Matplotlib |
| **Dataset** | 0% (downloaded) | MipNeRF360 (Google) |
| **3DGS Algorithm** | 0% (used) | Kerbl et al., 2023 |

---

## Learning Outcomes

Through this independent research project, I gained:

1. **HPC Systems:**
   - SLURM job scheduling and resource management
   - Multi-GPU allocation strategies
   - Environment configuration for distributed training

2. **Distributed Training:**
   - PyTorch DistributedDataParallel internals
   - Parameter synchronization mechanisms
   - Common failure modes and debugging

3. **3D Computer Vision:**
   - Neural rendering fundamentals
   - Gaussian splatting architecture
   - COLMAP structure-from-motion

4. **Research Methodology:**
   - Iterative experimental design
   - Root cause analysis
   - Technical problem documentation
   - Solution proposal development

5. **Software Engineering:**
   - Reproducible research practices
   - Comprehensive documentation
   - Version control and code organization

---

## Academic Integrity Statement

I certify that:

1. All experimental work was conducted by me on Iowa State's Nova cluster
2. All SLURM scripts and configuration files were written by me
3. All problem diagnosis and analysis represents my original work
4. All external tools and frameworks are properly cited
5. All contributions are clearly delineated from external dependencies

The research questions, methodology, execution, analysis, and documentation represent my independent work under the guidance of my advisors.

---

**Signature:** Mohammed Musthafa Rafi  
**Date:** December 2024  
**Course:** COMS 625 - Independent Study  
**Institution:** Iowa State University
