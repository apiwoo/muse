# Project MUSE - src/utils/cuda_helper.py
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import platform
import glob

def setup_cuda_environment():
    """
    [핵심] Windows 환경에서 DLL 로드 문제를 해결합니다.
    프로젝트 내부의 'libs' 폴더를 최우선으로 등록하여
    시스템 환경에 상관없이 안정적인 실행을 보장합니다.
    """
    if platform.system() != "Windows":
        return

    # print("🔧 [CUDA Helper] 라이브러리 경로 설정 중...")
    
    # Python site-packages 경로들 확인
    site_packages = [p for p in sys.path if 'site-packages' in p]
    
    dll_dirs = set()
    
    # [Custom Fix] 프로젝트 내부 'libs' 폴더 우선 추가 (Portable)
    # 현재 파일: src/utils/cuda_helper.py
    # 루트 경로: src/utils/../.. (즉, 프로젝트 루트)
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    local_lib_path = os.path.join(project_root, "libs")
    
    if os.path.exists(local_lib_path):
        dll_dirs.add(local_lib_path)

    # 1. 일반적인 NVIDIA 패키지 경로 탐색
    for sp in site_packages:
        nvidia_path = os.path.join(sp, "nvidia")
        if os.path.exists(nvidia_path):
            for root, dirs, files in os.walk(nvidia_path):
                if os.path.basename(root) in ['bin', 'lib']:
                    if any(f.endswith('.dll') for f in files):
                        dll_dirs.add(root)

        # 2. Torch 라이브러리
        torch_lib = os.path.join(sp, "torch", "lib")
        if os.path.exists(torch_lib):
            dll_dirs.add(torch_lib)

    # 3. 발견된 경로 등록
    found_cudnn_8 = False
    
    for directory in dll_dirs:
        try:
            # Python 3.8+ DLL 로드 허용
            os.add_dll_directory(directory)
            # PATH 환경변수 업데이트
            os.environ['PATH'] = directory + os.pathsep + os.environ['PATH']
            
            if glob.glob(os.path.join(directory, "cudnn64_8.dll")):
                found_cudnn_8 = True
                
        except Exception:
            pass

    # 4. 진단
    if found_cudnn_8:
        # print("   ✅ [OK] 필수 DLL(cudnn64_8.dll)이 로드되었습니다.")
        pass
    else:
        print("   ⚠️ [Warning] 'cudnn64_8.dll'을 찾지 못했습니다.")
        print(f"      -> 프로젝트 루트에 'libs' 폴더를 만들고 파일을 넣어주세요: {local_lib_path}")