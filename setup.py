import sys
import subprocess
import os
import pkg_resources

# ==============================================================================
# [Project MUSE] Environment Setup Script (v9.1 Safety First)
# 
# Target Hardware: NVIDIA RTX 3060 (12GB) or Higher
# Update:
# - v9.1: Added user confirmation (y/n) before uninstalling packages.
# - v9.0: Added 'CRITICAL_VERSIONS' dictionary for strict version control.
# - v9.0: Prevented auto-upgrade of Numpy/ONNX during massive package installs.
# ==============================================================================

# [중요] 버전이 민감한 패키지들은 여기서 통합 관리하며, 설치 시 강제 제약조건으로 사용됩니다.
CRITICAL_VERSIONS = {
    "numpy": "1.26.4",
    "onnx": "1.14.0",
    "onnxruntime-gpu": "1.16.0",
    "tensorrt": "10.0.1",
    "tensorrt-cu12": "10.0.1",
    "tensorrt-cu12-bindings": "10.0.1",
    "tensorrt-cu12-libs": "10.0.1",
}

def run_pip(args, description=None):
    """
    Run pip command with current python executable.
    Prints a description if provided.
    """
    if description:
        print(f"   ... {description}")
    
    cmd = [sys.executable, "-m", "pip"] + args
    # 명령어 로깅 (너무 길면 잘라서 보여줌)
    cmd_str = ' '.join(cmd)
    if len(cmd_str) > 200:
        print(f"   $ {cmd_str[:200]} ... (truncated)")
    else:
        print(f"   $ {cmd_str}")
        
    subprocess.check_call(cmd)

def get_installed_version(package_name):
    """현재 환경에 설치된 패키지의 버전을 반환합니다."""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def ask_user_permission(msg):
    """사용자에게 y/n 질문을 하고 동의 여부를 반환합니다."""
    while True:
        response = input(f"\n⚠️  {msg} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False

def uninstall_conflicts():
    print("\n🧹 [Step 1] Checking for conflicting packages...")
    
    # 충돌 가능성이 있는 패키지 목록
    uninstall_list = [
        "tensorrt", "tensorrt-cu12", "tensorrt-cu12-bindings", "tensorrt-cu12-libs",
        "tensorrt-libs", "onnx", "onnxruntime", "onnxruntime-gpu",
        "numpy" # Numpy도 일단 날리고 클린 설치하는 것이 안전함
    ]
    
    # 실제로 설치된 것만 골라내서 삭제 대상 확인
    targets = [pkg for pkg in uninstall_list if get_installed_version(pkg)]
    
    if targets:
        print(f"   Found conflicting packages: {', '.join(targets)}")
        
        # [v9.1 추가] 사용자에게 삭제 동의 구하기
        msg = "To ensure a clean installation, these packages need to be removed and re-installed.\n   Do you want to proceed with cleanup?"
        if not ask_user_permission(msg):
            print("   ⏭️  Skipping cleanup (Not recommended, but proceeding).")
            return

        try:
            run_pip(["uninstall", "-y"] + targets, description="Removing conflicts")
            print("   ✅ Cleanup complete.")
        except subprocess.CalledProcessError:
            print("   ⚠️ Cleanup skipped or failed.")
    else:
        print("   ✅ No conflicts found. Clean start!")

def install_numpy_base():
    """
    Numpy는 모든 라이브러리의 기초이므로 가장 먼저, 그리고 강력하게 고정합니다.
    """
    print(f"\n📦 [Step 2] Installing Base: Numpy == {CRITICAL_VERSIONS['numpy']}")
    run_pip(["install", f"numpy=={CRITICAL_VERSIONS['numpy']}"])

def install_pytorch_cuda():
    print("\n🔥 [Step 3] Verifying PyTorch CUDA 12.1 (cu121)...")
    
    # 이미 설치되어 있고 CUDA가 잡히면 스킵
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ PyTorch GPU version is active: {torch.__version__}")
            # 단, 버전이 너무 다르면 재설치 고려 가능하나 일단 유지
            return
    except ImportError:
        pass

    # PyTorch 설치 시에도 Numpy 다운그레이드/업그레이드 방지를 위해 numpy 버전 명시
    run_pip([
        "install",
        "torch", "torchvision", "torchaudio",
        f"numpy=={CRITICAL_VERSIONS['numpy']}",  # [방어 코드] PyTorch가 Numpy 바꾸지 못하게 함
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], description="Installing PyTorch (cu121)")

def install_core_dependencies():
    print("\n🚀 [Step 4] Installing Core & AI Dependencies...")
    
    # 1. 기본 의존성 리스트
    base_deps = [
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
        "nvidia-cudnn-cu12",
        "pygrabber",
        "timm",        
        "transformers",
        "git+https://github.com/facebookresearch/segment-anything-2.git"
    ]

    # Cupy 처리
    try:
        import cupy
    except ImportError:
        base_deps.append("cupy-cuda12x")

    # 2. [핵심 로직] 설치 명령어 구성 시 CRITICAL_VERSIONS를 '동시에' 설치 리스트에 넣습니다.
    # 이렇게 하면 pip resolver가 transformers 등을 설치할 때 
    # numpy나 onnx 버전을 멋대로 바꾸려다가 "사용자가 지정한 버전(Constraint)"에 막혀서
    # 호환되는 버전을 찾거나 에러를 뱉습니다 (제멋대로 바꾸는 것보다 에러가 낫습니다).
    
    final_install_list = base_deps.copy()
    
    # Critical Version들을 install list에 포함 (==버전 명시)
    for pkg, ver in CRITICAL_VERSIONS.items():
        # 이미 설치되었어도, 다시 한번 명시하여 '업그레이드 방지' 쐐기를 박음
        final_install_list.append(f"{pkg}=={ver}")

    try:
        # 한 번에 설치 (Resolver가 전체 의존성을 고려하도록 유도)
        run_pip(["install"] + final_install_list, description="Installing Main Dependencies with Constraints")
        print("\n   ✅ Core dependencies installed.")
    except subprocess.CalledProcessError:
        print("\n   ❌ Package installation failed during Step 4.")
        print("   💡 Tip: Check for version conflicts in the output above.")
        sys.exit(1)

def verify_install():
    """
    설치가 모두 끝난 후, 실제로 깔린 버전들이 CRITICAL_VERSIONS와 일치하는지 검사합니다.
    """
    print("\n🔎 [Step 5] Verifying Installation Integrity...")
    all_pass = True
    
    # 1. Critical List 검사
    for pkg, expected_ver in CRITICAL_VERSIONS.items():
        installed_ver = get_installed_version(pkg)
        if installed_ver == expected_ver:
            print(f"   ✅ {pkg:<20} : {installed_ver} (Matches)")
        else:
            print(f"   ❌ {pkg:<20} : {installed_ver} (Expected: {expected_ver})")
            all_pass = False
            
    # 2. PyTorch GPU 검사
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        print(f"   {'✅' if gpu_ok else '❌'} PyTorch CUDA      : {'Available' if gpu_ok else 'Not Available'}")
        if not gpu_ok: all_pass = False
    except ImportError:
        print("   ❌ PyTorch            : Not Installed")
        all_pass = False

    return all_pass

def main():
    print("============================================================")
    print("   Project MUSE - Environment Setup (v9.1 Safety First)")
    print("============================================================")
    
    # 1. 기존 충돌 패키지 정리 (사용자 확인 추가됨)
    uninstall_conflicts()

    # 2. Numpy 고정 설치
    install_numpy_base()

    # 3. PyTorch 설치 (Numpy 방어 포함)
    install_pytorch_cuda()

    # 4. 나머지 의존성 설치 (Critical Version 포함하여 방어)
    install_core_dependencies()

    # 5. 최종 검증
    if verify_install():
        print("\n============================================================")
        print("🎉 Setup Successfully Completed!")
        print("👉 Please run 'tools/download_models.py' next.")
        print("============================================================")
    else:
        print("\n============================================================")
        print("⚠️ Setup Completed with Warnings.")
        print("👉 Some versions do not match the target configuration.")
        print("👉 Check the '❌' marks above.")
        print("============================================================")

if __name__ == "__main__":
    main()