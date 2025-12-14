#!/usr/bin/env python3
"""
Setup Verification Script
=========================

Verifies that the environment is correctly configured for
distributed 3D Gaussian Splatting training.

Author: Mohammed Musthafa Rafi
Date: December 14 2025

Usage: python scripts/verify_setup.py
"""

import sys
import subprocess
from pathlib import Path

def check_header(text):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def check_import(module_name, display_name=None):
    """Check if a Python module can be imported"""
    if display_name is None:
        display_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {display_name:30s} OK")
        return True
    except ImportError as e:
        print(f"✗ {display_name:30s} FAILED: {e}")
        return False

def check_command(command, display_name):
    """Check if a command-line tool is available"""
    try:
        result = subprocess.run(
            [command, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✓ {display_name:30s} {version}")
            return True
        else:
            print(f"✗ {display_name:30s} Not found")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"✗ {display_name:30s} Not found")
        return False

def check_cuda():
    """Check CUDA availability and version"""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA {cuda_version} available")
            print(f"  └─ {gpu_count} GPU(s) detected: {gpu_name}")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except ImportError:
        print("✗ PyTorch not installed (can't check CUDA)")
        return False

def check_dataset():
    """Check if dataset is downloaded and processed"""
    dataset_path = Path("data/bicycle/processed")
    if dataset_path.exists():
        # Check for required files
        images_dir = dataset_path / "images"
        sparse_dir = dataset_path / "sparse"
        
        checks = {
            "Images directory": images_dir.exists(),
            "Sparse reconstruction": sparse_dir.exists(),
        }
        
        if sparse_dir.exists():
            checks["cameras.bin"] = (sparse_dir / "cameras.bin").exists() or \
                                    (sparse_dir / "0" / "cameras.bin").exists()
            checks["images.bin"] = (sparse_dir / "images.bin").exists() or \
                                   (sparse_dir / "0" / "images.bin").exists()
            checks["points3D.bin"] = (sparse_dir / "points3D.bin").exists() or \
                                     (sparse_dir / "0" / "points3D.bin").exists()
        
        all_good = all(checks.values())
        
        for item, status in checks.items():
            symbol = "✓" if status else "✗"
            print(f"{symbol} {item:30s} {'OK' if status else 'Missing'}")
        
        return all_good
    else:
        print(f"✗ Dataset not found at: {dataset_path}")
        print("  Run: bash scripts/setup_dataset.sh")
        return False

def main():
    """Run all verification checks"""
    
    print("Distributed 3DGS Environment Verification")
    print("Author: Mohammed Musthafa Rafi")
    print(f"Python: {sys.version.split()[0]}")
    
    all_checks_passed = True
    
    # Check Python version
    check_header("Python Version")
    if sys.version_info >= (3, 10):
        print(f"✓ Python {sys.version.split()[0]} (>= 3.10 required)")
    else:
        print(f"✗ Python {sys.version.split()[0]} (>= 3.10 required)")
        all_checks_passed = False
    
    # Check core dependencies
    check_header("Core Dependencies")
    checks = [
        check_import("torch", "PyTorch"),
        check_import("torchvision", "TorchVision"),
        check_import("numpy", "NumPy"),
        check_import("PIL", "Pillow"),
        check_import("cv2", "OpenCV"),
    ]
    all_checks_passed = all_checks_passed and all(checks)
    
    # Check Nerfstudio and 3DGS dependencies
    check_header("Nerfstudio & 3DGS")
    checks = [
        check_import("nerfstudio", "Nerfstudio"),
        check_import("gsplat", "gsplat"),
    ]
    all_checks_passed = all_checks_passed and all(checks)
    
    # Check command-line tools
    check_header("Command-Line Tools")
    checks = [
        check_command("ns-train", "Nerfstudio Training"),
        check_command("ns-render", "Nerfstudio Rendering"),
    ]
    all_checks_passed = all_checks_passed and all(checks)
    
    # Check CUDA
    check_header("CUDA Support")
    cuda_ok = check_cuda()
    all_checks_passed = all_checks_passed and cuda_ok
    
    # Check dataset
    check_header("Dataset")
    dataset_ok = check_dataset()
    all_checks_passed = all_checks_passed and dataset_ok
    
    # Check directory structure
    check_header("Project Structure")
    required_dirs = [
        "slurm_scripts",
        "scripts",
        "results",
        "figures",
    ]
    
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists():
            print(f"✓ {dir_name:30s} OK")
        else:
            print(f"✗ {dir_name:30s} Missing")
            all_checks_passed = False
    
    # Final summary
    check_header("Summary")
    if all_checks_passed:
        print("✓ All checks passed!")
        print("\nYou're ready to run training:")
        print("  sbatch slurm_scripts/train_1gpu.slurm")
        return 0
    else:
        print("✗ Some checks failed")
        print("\nPlease fix the issues above before running training.")
        print("\nCommon fixes:")
        print("  - Install dependencies: conda env create -f environment.yml")
        print("  - Download dataset: bash scripts/setup_dataset.sh")
        print("  - Load CUDA module: module load cuda/11.8")
        return 1

if __name__ == "__main__":
    sys.exit(main())
