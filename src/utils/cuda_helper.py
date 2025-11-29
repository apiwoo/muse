# Project MUSE - src/utils/cuda_helper.py
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import platform
import glob

def setup_cuda_environment():
    """
    [핵심] Windows 환경에서 pip로 설치된 NVIDIA 라이브러리(DLL)들을
    시스템 경로(PATH)에 강제로 주입합니다. (Final Version)
    """
    if platform.system() != "Windows":
        return

    print("🔧 [CUDA Helper] NVIDIA 라이브러리 경로 탐색 중...")
    
    # Python site-packages 경로들 확인
    site_packages = [p for p in sys.path if 'site-packages' in p]
    
    found_dlls = 0
    
    for sp in site_packages:
        # nvidia 폴더 내부 탐색
        nvidia_path = os.path.join(sp, "nvidia")
        if not os.path.exists(nvidia_path):
            continue
            
        # nvidia 폴더 하위의 모든 디렉토리 검사 (cudnn, cublas 등)
        # cuDNN 9.x는 보통 'nvidia/cudnn/bin' 또는 'nvidia/cudnn/lib'에 DLL이 있음
        for root, dirs, files in os.walk(nvidia_path):
            # 'bin' 또는 'lib' 폴더가 있으면 후보
            if os.path.basename(root) in ['bin', 'lib']:
                # DLL 파일이 하나라도 있는지 확인
                dlls = glob.glob(os.path.join(root, "*.dll"))
                if dlls:
                    try:
                        # Python 3.8+ 필수
                        os.add_dll_directory(root)
                        # 하위 호환성
                        os.environ['PATH'] = root + os.pathsep + os.environ['PATH']
                        # print(f"   ✅ DLL 경로 추가: {root}") 
                        found_dlls += 1
                    except Exception as e:
                        print(f"   ⚠️ 경로 등록 실패: {root} ({e})")

    if found_dlls > 0:
        print(f"   ✅ 총 {found_dlls}개의 NVIDIA 라이브러리 경로를 로드했습니다.")
    else:
        print("   ⚠️ [Warning] NVIDIA 라이브러리를 찾지 못했습니다. 'setup.py'를 확인하세요.")