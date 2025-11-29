import sys
import subprocess
import os

# ==============================================================================
# [Project MUSE] Environment Setup Script (v6.8 Final: cuDNN 9.x Restore)
# 
# Target Hardware: NVIDIA RTX 3060 (12GB) or Higher
# Core Philosophy: "Visual Supremacy on Mode A"
#
# v6.8 Update:
# - [Critical] onnxruntime-gpu 1.19+ requires cuDNN 9.x
# - Reverting to nvidia-cudnn-cu12 for modern compatibility
# ==============================================================================

def install_package_force(package_command):
    cmd = [sys.executable, "-m", "pip", "install"] + package_command
    print(f"📦 [Install] Installing packages...")
    try:
        subprocess.check_call(cmd)
        print(f"   ✅ Installation success")
    except subprocess.CalledProcessError:
        print(f"   ❌ Installation failed")

def install_pytorch_cuda():
    print("\n🔥 [System] Verifying PyTorch CUDA 12.1 (cu121)...")
    try:
        import torch
        if torch.cuda.is_available():
            print("   ✅ PyTorch GPU version is active. (Skipping re-install)")
            return
    except ImportError:
        pass

    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    subprocess.check_call(cmd)

def main():
    print("============================================================")
    print("   Project MUSE - Environment Setup (v6.8 cuDNN 9.x)")
    print("============================================================")
    
    # 1. Remove old cuDNN 8.x (Prevent DLL conflict)
    print("🗑️ Removing old cuDNN 8.x...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "nvidia-cudnn-cu11", "nvidia-cublas-cu11"])
    except:
        pass

    # 2. Fix Numpy
    install_package_force(["numpy==1.26.4"])

    # 3. Install PyTorch
    install_pytorch_cuda()

    # 4. Install Dependencies (cuDNN 9.x included)
    print("\n🚀 [Step 2] Installing dependencies with cuDNN 9.x...")
    
    dependency_list = [
        "scipy", "pyyaml", "tqdm",
        "cupy-cuda12x",   
        "opencv-python<4.11", 
        "mediapipe",      
        "insightface",    
        "tensorrt",       
        "onnx", 
        "onnxruntime-gpu",
        "moderngl",       
        "moderngl-window",
        "pyvirtualcam",   
        "imgui",          
        "PySide6", 
        "pyqtdarktheme",
        
        # [Restore] Use cuDNN 9.x for latest onnxruntime
        "nvidia-cudnn-cu12" 
    ]

    try:
        import cupy
        if "cupy-cuda12x" in dependency_list:
            dependency_list.remove("cupy-cuda12x")
    except ImportError:
        pass

    install_package_force(dependency_list)

    print("\n============================================================")
    print("🎉 Setup Complete.")
    print("👉 Please run 'src/utils/cuda_helper.py' logic is updated next.")
    print("============================================================")

if __name__ == "__main__":
    main()