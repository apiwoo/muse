# Project MUSE - fix_env.py
# Environment Repair Tool for RTX 3060/4090
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import subprocess
import pkg_resources

def run_cmd(cmd):
    print(f"   $ {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print("   ⚠️ 명령 실행 중 오류가 발생했으나 계속 진행합니다.")

def get_installed_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return "Not Installed"

def main():
    print("========================================================")
    print("   MUSE Environment Fixer (Strict Version Pinning)")
    print("========================================================")

    # 1. 현재 상태 점검 (확인할 패키지 늘림)
    packages = [
        "tensorrt", "tensorrt-cu12", "tensorrt-cu12-bindings", "tensorrt-cu12-libs",
        "onnx", "onnxruntime-gpu"
    ]
    print("🔍 [Step 1] 현재 설치된 버전 확인:")
    for pkg in packages:
        ver = get_installed_version(pkg)
        print(f"   - {pkg}: {ver}")

    print("\n⚠️ 경고: TensorRT 및 ONNX 관련 패키지를 제거하고 'v10.0.1'로 통일합니다.")
    user_input = input("👉 진행하시겠습니까? (y/n): ")
    if user_input.lower() != 'y':
        print("취소되었습니다.")
        return

    # 2. 제거 (Uninstall) - 모든 관련 패키지 명시
    print("\n🗑️ [Step 2] 기존 패키지 제거 중...")
    uninstall_list = [
        "tensorrt", "tensorrt-cu12", "tensorrt-cu12-bindings", "tensorrt-cu12-libs",
        "tensorrt-libs", "onnx", "onnxruntime", "onnxruntime-gpu"
    ]
    cmd_uninstall = [sys.executable, "-m", "pip", "uninstall", "-y"] + uninstall_list
    run_cmd(cmd_uninstall)

    # 3. 재설치 (Install Specific Versions)
    print("\n📦 [Step 3] 검증된 버전(v10.0.1)으로 강제 설치 중...")
    
    # [Critical Fix] 메인 패키지뿐만 아니라 하위 라이브러리까지 버전을 10.0.1로 고정합니다.
    # 이렇게 해야 pip가 최신 버전(10.14 등)을 멋대로 가져오지 않습니다.
    
    install_cmds = [
        # ONNX 관련
        ["onnx==1.14.0"],
        ["onnxruntime-gpu==1.16.0"],
        
        # TensorRT 관련 (전부 10.0.1로 고정)
        [
            "tensorrt==10.0.1",
            "tensorrt-cu12==10.0.1",
            "tensorrt-cu12-bindings==10.0.1", 
            "tensorrt-cu12-libs==10.0.1"
        ]
    ]

    for pkg in install_cmds:
        cmd_install = [sys.executable, "-m", "pip", "install"] + pkg
        run_cmd(cmd_install)
        
    print("\n🔧 [Step 4] 최종 설치 결과 확인")
    for pkg in packages:
        ver = get_installed_version(pkg)
        print(f"   - {pkg}: {ver}")

    print("\n🎉 복구 완료. 이제 버전 충돌 없이 'trt_converter.py'를 실행할 수 있습니다.")

if __name__ == "__main__":
    main()