# Project MUSE - collect_libs.py
# (C) 2025 MUSE Corp. All rights reserved.
# 역할: 배포(Portable)를 위해 흩어진 NVIDIA 핵심 DLL들과 SAM 2 설정 파일들을 수집합니다.

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

def collect_sam2_configs(project_root):
    """
    SAM 2 라이브러리 내부의 configs 폴더를 찾아서
    프로젝트의 assets/sam2_configs 로 복사합니다.
    """
    print("\n🔍 [SAM 2 Config] 설정 파일 수집 시작...")
    
    # 1. 타겟 경로 (프로젝트 내부)
    target_dir = os.path.join(project_root, "assets", "sam2_configs")
    
    # 2. 소스 경로 찾기 (라이브러리 import 이용)
    try:
        import sam2
        sam2_pkg_root = os.path.dirname(sam2.__file__)
        
        # 가능한 소스 경로 후보들
        candidates = [
            os.path.join(sam2_pkg_root, "configs"), # pip 일반 설치
            os.path.join(os.path.dirname(sam2_pkg_root), "sam2_configs"), # 일부 변종 설치
            os.path.join(os.path.dirname(sam2_pkg_root), "configs") # 소스 설치
        ]
        
        source_dir = None
        for path in candidates:
            if os.path.exists(path) and os.path.isdir(path):
                # 유효성 검사 (yaml 파일이 있는지)
                # [Fix] 괄호 위치 수정: recursive=True는 glob.glob의 인자여야 함
                if glob.glob(os.path.join(path, "*.yaml")) or glob.glob(os.path.join(path, "**/*.yaml"), recursive=True):
                    source_dir = path
                    break
        
        if source_dir:
            print(f"   -> 원본 발견: {source_dir}")
            
            # 기존 폴더가 있으면 삭제 후 다시 복사 (최신화)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            
            # recursive 인자는 제거됨 (shutil.copytree는 기본적으로 재귀적)
            shutil.copytree(source_dir, target_dir)
            print(f"   ✅ 복사 완료: {target_dir}")
            return True
        else:
            print("   ⚠️ SAM 2 Config 폴더를 찾지 못했습니다. (라이브러리 설치 상태 확인 필요)")
            return False
            
    except ImportError:
        print("   ❌ 'sam2' 모듈을 import할 수 없습니다. 설치되어 있나요?")
        return False
    except Exception as e:
        print(f"   ❌ 복사 중 오류 발생: {e}")
        return False

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

    # 2. SAM 2 Config 수집 (추가된 로직)
    collect_sam2_configs(project_root)

    print("\n🔍 [NVIDIA DLL] Scanning site-packages...")

    # 3. 검색할 패키지 및 DLL 패턴 정의
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

    # 4. 결과 리포트
    print("-" * 60)
    if total_copied > 0:
        print(f"🎉 총 {total_copied}개의 핵심 DLL을 'libs' 폴더로 수집했습니다.")
        print("💡 이제 이 프로젝트 폴더를 통째로 다른 PC로 옮겨도 GPU 가속이 작동할 확률이 매우 높습니다.")
    else:
        print("ℹ️  새로 복사된 DLL 파일이 없습니다.")
        print("   -> 'libs' 폴더에 cudnn64_8.dll, nvinfer.dll 등이 있는지 확인하세요.")

if __name__ == "__main__":
    main()