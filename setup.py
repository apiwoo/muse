import sys
import subprocess
import os

# ==============================================================================
# [Project MUSE] Environment Setup Script (v7.1 Integrated: Camera Name Support)
# 
# Target Hardware: NVIDIA RTX 3060 (12GB) or Higher
# Core Philosophy: "Visual Supremacy on Mode A"
#
# v7.1 Update:
# - Added 'pygrabber' for Windows Camera Device Name Detection.
# - Merged 'tools/fix_env.py' logic into setup.
# - Enforced strict version pinning for TensorRT (10.0.1) & ONNXRuntime (1.16.0).
# - Added 'segment-anything' for Teacher Model Pipeline.
# ==============================================================================

def run_pip(args):
    """Run pip command with current python executable"""
    cmd = [sys.executable, "-m", "pip"] + args
    print(f"   $ {' '.join(cmd)}")
    subprocess.check_call(cmd)

def uninstall_conflicts():
    print("\n🧹 [Step 1] Cleaning up conflicting packages (Fix_Env Logic)...")
    # fix_env.py에서 정의한 충돌 유발 패키지 리스트
    uninstall_list = [
        "tensorrt", "tensorrt-cu12", "tensorrt-cu12-bindings", "tensorrt-cu12-libs",
        "tensorrt-libs", "onnx", "onnxruntime", "onnxruntime-gpu"
    ]
    try:
        # -y 옵션으로 묻지 않고 삭제
        run_pip(["uninstall", "-y"] + uninstall_list)
        print("   ✅ Cleanup complete.")
    except subprocess.CalledProcessError:
        print("   ⚠️ Cleanup skipped or failed (might be already clean).")

def install_pytorch_cuda():
    print("\n🔥 [Step 2] Verifying PyTorch CUDA 12.1 (cu121)...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ PyTorch GPU version is active: {torch.__version__}")
            return
    except ImportError:
        pass

    run_pip([
        "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ])

def main():
    print("============================================================")
    print("   Project MUSE - Environment Setup (v7.1 Final)")
    print("============================================================")
    
    # 1. 기존 충돌 패키지 정리 (fix_env.py 로직 반영)
    uninstall_conflicts()

    # 2. Numpy 고정 (호환성 이슈 방지)
    print("\n📦 [Step 3] Installing Base Dependencies...")
    run_pip(["install", "numpy==1.26.4"])

    # 3. PyTorch 설치
    install_pytorch_cuda()

    # 4. 의존성 설치 (Strict Version Pinning)
    print("\n🚀 [Step 4] Installing Core & AI Dependencies...")
    
    dependency_list = [
        "scipy", "pyyaml", "tqdm",
        "opencv-python<4.11", 
        "mediapipe",      
        "insightface",    
        "moderngl",       
        "moderngl-window",
        "pyvirtualcam",   
        "imgui",          
        "PySide6", 
        "pyqtdarktheme",
        "nvidia-cudnn-cu12", # cuDNN 9.x Support
        
        # [New] Windows Camera Name Detection (For Recorder)
        "pygrabber",

        # [Fix_Env Reflection] Strict Versions for Stability
        # fix_env.py에서 검증된 버전들을 강제합니다.
        "onnx==1.14.0",
        "onnxruntime-gpu==1.16.0",
        "tensorrt==10.0.1",
        "tensorrt-cu12==10.0.1",
        "tensorrt-cu12-bindings==10.0.1", 
        "tensorrt-cu12-libs==10.0.1",

        # [New] Teacher Model (SAM)
        "git+https://github.com/facebookresearch/segment-anything.git"
    ]

    # CuPy는 환경에 따라 자동 감지 설치 권장
    try:
        import cupy
    except ImportError:
        dependency_list.append("cupy-cuda12x")

    try:
        run_pip(["install"] + dependency_list)
        print("\n   ✅ All packages installed successfully.")
    except subprocess.CalledProcessError:
        print("\n   ❌ Package installation failed.")
        sys.exit(1)

    print("\n============================================================")
    print("🎉 Setup Complete.")
    print("👉 Please run 'cm.py' to update project structure.")
    print("============================================================")

if __name__ == "__main__":
    main()