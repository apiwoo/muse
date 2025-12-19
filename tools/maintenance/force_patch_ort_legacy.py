# Project MUSE - force_patch_ort.py
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import shutil
import glob
import site

def get_site_packages():
    """Python site-packages 경로를 찾습니다."""
    return site.getsitepackages()[0]

def find_ort_capi_dirs(site_pkg):
    """
    site-packages 내에서 onnxruntime 관련 패키지 폴더를 찾고,
    그 내부의 capi 폴더 경로들을 반환합니다.
    """
    ort_dirs = []
    # onnxruntime으로 시작하는 모든 디렉토리 검색 (onnxruntime, onnxruntime_gpu 등)
    # 윈도우에서는 대소문자 구분이 없을 수 있지만 glob은 패턴 매칭을 수행합니다.
    candidates = glob.glob(os.path.join(site_pkg, "onnxruntime*"))
    
    for d in candidates:
        if os.path.isdir(d):
            capi_path = os.path.join(d, "capi")
            if os.path.exists(capi_path):
                ort_dirs.append(capi_path)
    
    return ort_dirs

def main():
    print("========================================================")
    print("   MUSE ONNXRuntime Force Patcher (In-Place Injection)")
    print("========================================================")

    site_pkg = get_site_packages()
    print(f"📂 Site-Packages: {site_pkg}")

    # 1. 타겟 설정: onnxruntime*/capi (유연한 검색)
    target_dirs = find_ort_capi_dirs(site_pkg)
    
    if not target_dirs:
        print("❌ onnxruntime 설치 경로를 찾을 수 없습니다.")
        print("   (pip list로 onnxruntime 또는 onnxruntime-gpu가 설치되어 있는지 확인하세요)")
        return

    print(f"🎯 Target Injection Paths ({len(target_dirs)} found):")
    for d in target_dirs:
        print(f"   - {d}")

    # 2. 소스 설정: nvidia 패키지들
    nvidia_dir = os.path.join(site_pkg, "nvidia")
    if not os.path.exists(nvidia_dir):
        print("❌ nvidia 패키지 경로를 찾을 수 없습니다.")
        print("   (pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 등이 필요합니다)")
        return

    # 3. 복사할 DLL 패턴
    # cudnn, cublas, curand, cufft 등 모든 핵심 DLL
    dll_patterns = [
        "**/bin/*.dll", # nvidia 패키지 구조상 bin 안에 dll이 있음
        "**/lib/*.dll"
    ]

    total_count = 0
    print("\n🚀 Injecting DLLs into ONNXRuntime...")
    
    # nvidia 폴더 하위의 모든 DLL을 검색해서 각 타겟 디렉토리로 복사
    for root, dirs, files in os.walk(nvidia_dir):
        for file in files:
            if file.endswith(".dll"):
                src_path = os.path.join(root, file)
                
                for target_dir in target_dirs:
                    dst_path = os.path.join(target_dir, file)
                    
                    # 이미 있어도 덮어쓰기 (버전 꼬임 방지)
                    try:
                        shutil.copy2(src_path, dst_path)
                        # print(f"   -> Injected: {file} into {os.path.basename(os.path.dirname(target_dir))}") 
                        total_count += 1
                    except Exception as e:
                        print(f"   ⚠️ Copy Failed: {file} -> {target_dir} ({e})")

    if total_count > 0:
        print(f"\n🎉 성공! 총 {total_count}번의 파일 복사가 수행되었습니다.")
        print("👉 이제 무조건 실행될 것입니다. 'python src/main.py'를 실행하세요.")
    else:
        print("\n❌ 복사할 DLL을 찾지 못했습니다. 'pip install nvidia-cudnn-cu12' 등이 제대로 설치되었나요?")

if __name__ == "__main__":
    main()