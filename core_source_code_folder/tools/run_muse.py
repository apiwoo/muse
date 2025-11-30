# Project MUSE - run_muse.py
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import glob
import subprocess
import site

def find_nvidia_dll_paths():
    """
    Python site-packages 내의 nvidia 관련 모든 패키지들(cudnn, cublas 등)의
    DLL이 들어있는 폴더 경로를 광범위하게 찾습니다.
    """
    dll_paths = set() # 중복 제거를 위해 set 사용
    
    # site-packages 경로 찾기
    site_packages_list = site.getsitepackages()
    # 사용자 site-packages도 추가 (AppData 등)
    user_site = site.getusersitepackages()
    if os.path.exists(user_site):
        site_packages_list.append(user_site)
    
    print(f"🔍 Searching in site-packages: {site_packages_list}")

    for sp in site_packages_list:
        if not os.path.exists(sp): continue

        # 1. 'nvidia' 폴더 내부 검색 (일반적인 경우)
        nvidia_root = os.path.join(sp, "nvidia")
        if os.path.exists(nvidia_root):
            for root, dirs, files in os.walk(nvidia_root):
                if any(f.endswith('.dll') for f in files):
                    dll_paths.add(root)

        # 2. 'nvidia_*' 패키지 폴더 검색 (예: nvidia_cudnn_cu12)
        # onnxruntime-gpu는 주로 cudnn, cublas 관련 dll을 찾습니다.
        target_patterns = ["nvidia_cudnn*", "nvidia_cublas*", "nvidia_cufft*", "nvidia_curand*"]
        
        for pattern in target_patterns:
            for pkg_dir in glob.glob(os.path.join(sp, pattern)):
                if os.path.isdir(pkg_dir):
                    # 패키지 폴더 내부 탐색 (bin, lib, 또는 루트)
                    for root, dirs, files in os.walk(pkg_dir):
                        if any(f.endswith('.dll') for f in files):
                            dll_paths.add(root)
                        
    return list(dll_paths)

def main():
    print("========================================================")
    print("   MUSE Launcher (Enhanced Auto Environment Config)")
    print("========================================================")

    # 1. NVIDIA 라이브러리 경로 찾기
    nvidia_paths = find_nvidia_dll_paths()
    
    if not nvidia_paths:
        print("⚠️ Warning: NVIDIA 패키지 경로를 찾지 못했습니다.")
        print("   (pip install nvidia-cudnn-cu12 등이 설치되어 있는지 확인하세요)")
    else:
        print(f"✅ Found {len(nvidia_paths)} NVIDIA library paths.")
        for p in nvidia_paths:
            print(f"   -> {p}")

    # 2. 환경 변수 PATH 업데이트 (현재 프로세스 및 자식 프로세스용)
    current_path = os.environ.get('PATH', '')
    # 우선순위를 위해 nvidia 경로들을 맨 앞에 배치
    new_path = os.pathsep.join(nvidia_paths) + os.pathsep + current_path
    
    # 환경 변수 딕셔너리 복사 및 업데이트
    env = os.environ.copy()
    env['PATH'] = new_path
    
    # [Critical] ONNXRuntime 및 TensorRT를 위한 환경변수 설정
    # LD_LIBRARY_PATH는 리눅스용이지만 일부 라이브러리가 참고할 수 있음
    env['LD_LIBRARY_PATH'] = new_path 
    
    for p in nvidia_paths:
        # cuDNN 및 cuBLAS 경로 명시 (일부 구버전 ORT 대응)
        if 'cudnn' in p.lower():
            env['CUDNN_PATH'] = p
        if 'cublas' in p.lower():
            env['CUBLAS_PATH'] = p

    # 3. 메인 프로그램 실행
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(project_root, "src", "main.py")
    
    print("-" * 60)
    print(f"🚀 Launching MUSE: {main_script}")
    print("-" * 60)
    
    try:
        # 서브프로세스로 main.py 실행 (업데이트된 환경 변수 전달)
        subprocess.run([sys.executable, main_script], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ MUSE 실행 중 오류 발생: {e}")
    except KeyboardInterrupt:
        print("\n🛑 종료합니다.")

if __name__ == "__main__":
    main()