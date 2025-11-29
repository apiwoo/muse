import sys
import subprocess
import os

# ==============================================================================
# [Project MUSE] Environment Setup Script (v6.3 Final: User-Proven Fix)
# 
# Target Hardware: NVIDIA RTX 3060 (12GB) or Higher
# Core Philosophy: "Visual Supremacy on Mode A"
#
# v6.3 Update:
# - [Solution] 사용자 경험 기반의 확실한 PyTorch CUDA 설치법 적용
# - --index-url https://download.pytorch.org/whl/cu121 사용
# - numpy==1.26.4 고정
# ==============================================================================

def install_package_force(package_command):
    """
    일반 패키지를 설치합니다.
    """
    cmd = [sys.executable, "-m", "pip", "install"] + package_command
    display_cmd = " ".join(package_command)
    print(f"📦 [Install] {display_cmd}")
    
    try:
        subprocess.check_call(cmd)
        print(f"   ✅ 설치 성공")
    except subprocess.CalledProcessError:
        print(f"   ❌ 설치 실패 (수동 확인 필요): {display_cmd}")

def install_pytorch_cuda():
    """
    [핵심] 사용자가 검증한 방식으로 PyTorch(cu121)를 설치합니다.
    """
    print("\n🔥 [System] PyTorch CUDA 12.1 (cu121) 강제 설치 루틴...")
    
    # 1. 기존에 잘못 깔린(CPU 버전 등) PyTorch 제거
    print("   🗑️ 충돌 방지를 위해 기존 PyTorch 관련 패키지를 제거합니다...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "-y", 
            "torch", "torchvision", "torchaudio"
        ])
    except Exception:
        pass

    # 2. 검증된 명령어로 설치 (YOLO 프로젝트 방식 적용)
    # 최신 호환 버전을 가져오되, 인덱스는 무조건 cu121을 바라보게 함
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    print(f"   🚀 [Execute] pip install ... --index-url https://download.pytorch.org/whl/cu121")
    print("   ⏳ 다운로드 용량이 큽니다 (약 2~3GB). 잠시만 기다려주세요...")
    
    try:
        subprocess.check_call(cmd)
        print("   ✅ PyTorch CUDA 버전 설치 완료!")
    except subprocess.CalledProcessError:
        print("   ❌ 설치 실패. 인터넷 연결을 확인하거나, 터미널에서 위 명령어를 직접 실행해보세요.")

def check_gpu_compatibility():
    """
    최종 GPU 호환성 점검
    """
    print("\n🔍 최종 시스템 점검 (Mode A)...")
    try:
        import torch
        # 재로드
        import importlib
        importlib.reload(torch)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            cuda_ver = torch.version.cuda
            
            print(f"   ✅ GPU 인식 성공: {gpu_name} ({vram_gb:.1f} GB)")
            print(f"   ✅ CUDA Runtime: {cuda_ver}")
            
            print("\n   🚀 [Mission Complete] 모든 준비가 끝났습니다.")
            print("       이제 'main.py'를 실행하면 RTX 3060이 불을 뿜을 것입니다.")
        else:
            print("   ❌ 여전히 CUDA를 사용할 수 없습니다.")
            print("      이유: Python 인터프리터가 GPU를 못 찾고 있습니다.")
            
    except ImportError:
        print("   ⚠️ PyTorch 임포트 실패.")

def main():
    print("============================================================")
    print("   Project MUSE - Environment Setup (v6.3 Final)")
    print("============================================================")
    
    # 1. Numpy 우선 고정 (호환성 1순위)
    print("\n🚀 [Step 1] Numpy 버전 고정 (v1.26.4)...")
    install_package_force(["numpy==1.26.4"])

    # 2. PyTorch CUDA 설치 (가장 중요)
    install_pytorch_cuda()

    # 3. 나머지 필수 패키지
    print("\n🚀 [Step 2] 나머지 의존성 패키지 설치...")
    other_packages = [
        "scipy", "pyyaml", "tqdm",
        "cupy-cuda12x",   
        "opencv-python",
        "mediapipe",      
        "insightface",    
        "tensorrt",       
        "onnx", "onnxruntime-gpu",
        "moderngl",       
        "moderngl-window",
        "pyvirtualcam",   
        "imgui",          
        "PySide6", 
        "pyqtdarktheme"
    ]

    for pkg in other_packages:
        if pkg.startswith("cupy"):
            try:
                import cupy
                print(f"   ✅ CuPy 이미 설치됨")
                continue
            except ImportError:
                pass
        install_package_force([pkg, "--upgrade"])

    # 4. 최종 점검
    check_gpu_compatibility()

    print("\n============================================================")
    print("🎉 설정 완료.")
    print("👉 이제 'main.py'를 실행하여 FPS 60이 나오는지 확인하세요!")
    print("============================================================")

if __name__ == "__main__":
    main()