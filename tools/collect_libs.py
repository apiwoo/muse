# Project MUSE - collect_libs.py
# (C) 2025 MUSE Corp. All rights reserved.
# 역할: 배포(Portable)를 위해 흩어진 NVIDIA 핵심 DLL들을 libs 폴더로 수집합니다.

import os
import sys
import shutil
import glob
import site

def get_site_packages():
    """Python site-packages 경로를 찾습니다."""
    # 사용자별 경로와 시스템 경로 모두 확인
    paths = site.getsitepackages()
    user_site = site.getusersitepackages()
    if os.path.exists(user_site):
        paths.append(user_site)
    return paths

def main():
    print("========================================================")
    print("   MUSE Dependency Collector (Portable Builder)")
    print("========================================================")

    # 1. 타겟 폴더 설정 (프로젝트 루트/libs)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    libs_dir = os.path.join(project_root, "libs")
    
    if not os.path.exists(libs_dir):
        os.makedirs(libs_dir)
        print(f"📂 'libs' 폴더 생성됨: {libs_dir}")
    else:
        print(f"📂 Target Directory: {libs_dir}")

    # 2. 검색할 패키지 및 DLL 패턴 정의
    # 여기에 정의된 파일들이 없으면 실행 시 'DLL Load Failed'가 뜹니다.
    search_targets = [
        # (패키지 폴더명 키워드, [DLL 패턴 리스트])
        ("nvidia", [
            "**/cudnn*.dll",       # cuDNN (Deep Learning Core)
            "**/cublas*.dll",      # cuBLAS (Matrix Math)
            "**/cufft*.dll",       # cuFFT
            "**/curand*.dll",      # cuRAND
            "**/cusparse*.dll",    # cuSPARSE
            "**/cuda_runtime*.dll", # cudart
            "**/nvrtc*.dll"        # Runtime Compiler
        ]),
        ("tensorrt", [
            "**/nvinfer*.dll",     # TensorRT Core
            "**/nvonnxparser*.dll",# ONNX Parser
            "**/nvinfer_plugin*.dll"
        ]),
        ("torch", [
            "**/lib/torch_python.dll", # (Optional)
            "**/lib/c10_cuda.dll",
            "**/lib/c10.dll",
            "**/lib/torch_cpu.dll",
            "**/lib/torch_cuda.dll"
        ])
    ]

    site_paths = get_site_packages()
    print(f"🔍 Scanning site-packages: {len(site_paths)} locations")

    total_copied = 0
    
    for sp in site_paths:
        if not os.path.exists(sp): continue
        
        for pkg_keyword, patterns in search_targets:
            # 패키지 폴더 찾기 (예: nvidia-cudnn-cu12 등)
            # glob으로 해당 키워드가 포함된 폴더를 모두 찾음
            pkg_dirs = glob.glob(os.path.join(sp, f"*{pkg_keyword}*"))
            
            for pkg_dir in pkg_dirs:
                if not os.path.isdir(pkg_dir): continue
                
                # DLL 패턴 검색
                for pattern in patterns:
                    # recursive=True로 하위 폴더(bin, lib 등)까지 뒤짐
                    found_dlls = glob.glob(os.path.join(pkg_dir, pattern), recursive=True)
                    
                    for dll_path in found_dlls:
                        filename = os.path.basename(dll_path)
                        dst_path = os.path.join(libs_dir, filename)
                        
                        # 이미 존재하면 크기 비교 (더 큰 놈이 보통 정품(?)임)
                        should_copy = True
                        if os.path.exists(dst_path):
                            src_size = os.path.getsize(dll_path)
                            dst_size = os.path.getsize(dst_path)
                            if src_size == dst_size:
                                should_copy = False # 이미 같은 파일 있음
                        
                        if should_copy:
                            try:
                                shutil.copy2(dll_path, dst_path)
                                print(f"   -> Copied: {filename} ({os.path.getsize(dll_path)/1024/1024:.1f} MB)")
                                total_copied += 1
                            except Exception as e:
                                print(f"   ❌ Copy Failed: {filename} ({e})")

    # 3. 결과 리포트
    print("-" * 60)
    if total_copied > 0:
        print(f"🎉 총 {total_copied}개의 핵심 DLL을 'libs' 폴더로 수집했습니다.")
        print("💡 이제 이 프로젝트 폴더를 통째로 다른 PC로 옮겨도 GPU 가속이 작동할 확률이 매우 높습니다.")
    else:
        print("ℹ️  새로 복사된 파일이 없습니다. (이미 다 있거나, 패키지를 못 찾음)")
        print("   -> 'libs' 폴더에 cudnn64_8.dll, nvinfer.dll 등이 있는지 확인하세요.")

if __name__ == "__main__":
    main()