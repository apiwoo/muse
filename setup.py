import sys
import subprocess

# ==========================================
# [Project MUSE] 최신 환경 강제 업데이트 스크립트 (v4.1 - Lightweight)
# 목표: PyTorch를 제거하고 MediaPipe와 렌더링 엔진만 설치하여 가볍고 빠른 환경을 구축합니다.
# ==========================================

def install_package_force(package_command):
    """
    패키지를 강제로 업그레이드/설치합니다.
    """
    cmd = [sys.executable, "-m", "pip", "install"] + package_command
    
    # 보기 좋게 명령어 출력
    display_cmd = " ".join(package_command)
    print(f"📦 설치 시작: {display_cmd}")
    
    try:
        subprocess.check_call(cmd)
        print(f"   ✅ 설치 성공")
    except subprocess.CalledProcessError:
        print(f"   ❌ 설치 실패: {display_cmd}")
        # 필수 라이브러리 실패 시 경고 (치명적이지 않으면 진행)

def main():
    print("========================================")
    print("   Project MUSE - Environment Update (Lightweight)")
    print("========================================")
    
    print("\n🚀 AI 엔진(MediaPipe) 및 필수 라이브러리 설치를 시작합니다...")

    # 1. PyTorch 제거 (최적화: MediaPipe는 Torch가 필요 없음)
    # 기존 코드에서 Torch 설치 부분을 삭제하여 배포 용량을 1GB -> 200MB 수준으로 줄였습니다.

    # 2. Project MUSE 필수 라이브러리
    # numpy, opencv, mediapipe, pyqt6, moderngl 등 핵심만 설치
    required_packages = [
        "opencv-python",
        "numpy",
        "scipy",
        "pyvirtualcam",   # OBS 연동
        "PyQt6",          # UI
        "moderngl",       # 렌더링 (OpenGL)
        "pyyaml",
        "mediapipe",      # 구글 AI 엔진 (Face Mesh)
        "Cython"          # [추가] 상용화 시 보안(코드 컴파일) 목적
    ]

    print("\n🚀 필수 패키지 업데이트 중...")
    for pkg in required_packages:
        install_package_force([pkg, "--upgrade"])

    print("\n========================================")
    print("🎉 모든 업데이트가 완료되었습니다!")
    print("👉 PyTorch 없이 가볍게 구동됩니다.")
    print("👉 이제 'cm.py'를 실행하여 프로젝트 폴더를 생성하세요.")
    print("========================================")

if __name__ == "__main__":
    main()