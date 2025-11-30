# Project MUSE - patch_dll.py
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import shutil
import glob

def find_nvidia_packages():
    """Python site-packages 내의 nvidia 관련 패키지 경로를 찾습니다."""
    site_packages = [p for p in sys.path if 'site-packages' in p]
    nvidia_paths = []
    for sp in site_packages:
        nv_path = os.path.join(sp, "nvidia")
        if os.path.exists(nv_path):
            nvidia_paths.append(nv_path)
    return nvidia_paths

def main():
    print("========================================================")
    print("   MUSE DLL Patcher (Fix ONNXRuntime Error 126)")
    print("========================================================")

    # 1. 타겟 경로 설정 (현재 실행 위치 또는 site-packages의 onnxruntime/capi)
    # 가장 확실한 방법: 메인 실행 파일이 있는 곳(src) 또는 프로젝트 루트에 DLL을 다 쏟아붓는 것보다는
    # 시스템 PATH에 추가하거나, 필요한 곳에 심볼릭 링크를 거는 것이지만
    # 윈도우에서는 '복사'가 가장 확실합니다.
    
    # 여기서는 '프로젝트 루트'에 DLL을 복사하여 실행 시 바로 찾게 합니다.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"📂 Target Directory: {project_root}")

    # 2. NVIDIA 패키지 탐색
    nvidia_roots = find_nvidia_packages()
    if not nvidia_roots:
        print("❌ 'nvidia' 패키지를 찾을 수 없습니다. (pip install nvidia-cudnn-cu12 등 필요)")
        return

    # 3. 주요 DLL 복사 (cudnn, cublas, cufft 등)
    # onnxruntime-gpu가 필요로 하는 핵심 파일들
    dll_patterns = [
        "cudnn*/bin/*.dll",      # cuDNN
        "cublas*/bin/*.dll",     # cuBLAS
        "cufft*/bin/*.dll",      # cuFFT
        "curand*/bin/*.dll",     # cuRAND
        "cuda_runtime*/bin/*.dll" # cudart
    ]

    count = 0
    for nv_root in nvidia_roots:
        print(f"🔍 Scanning in: {nv_root}")
        for pattern in dll_patterns:
            # glob search
            search_path = os.path.join(nv_root, pattern)
            found_dlls = glob.glob(search_path)
            
            for dll_path in found_dlls:
                filename = os.path.basename(dll_path)
                target_path = os.path.join(project_root, filename)
                
                # 이미 있으면 스킵 (용량/수정시간 비교는 생략하고 단순 존재 여부만 체크)
                if not os.path.exists(target_path):
                    try:
                        shutil.copy2(dll_path, target_path)
                        print(f"   -> Copied: {filename}")
                        count += 1
                    except Exception as e:
                        print(f"   ❌ Copy Failed: {filename} ({e})")
    
    if count == 0:
        print("\nℹ️  새로 복사된 파일이 없습니다. (이미 존재하거나 파일을 못 찾음)")
    else:
        print(f"\n🎉 총 {count}개의 DLL 파일을 프로젝트 루트로 복사했습니다.")
    
    print("👉 이제 'python src/main.py'를 다시 실행해 보세요.")

if __name__ == "__main__":
    main()