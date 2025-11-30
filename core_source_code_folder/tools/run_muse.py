# Project MUSE - run_muse.py
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import glob
import subprocess
import site

def find_nvidia_dll_paths():
    """
    Python site-packages 내의 nvidia 관련 패키지와
    프로젝트 내장 'libs' 폴더를 탐색합니다.
    """
    dll_paths = set()
    
    # [정석 해결법] 프로젝트 내부의 'libs' 폴더를 최우선으로 탐색
    # 이렇게 하면 외부 가상환경에 의존하지 않고 독립적으로 실행 가능합니다.
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    local_lib_path = os.path.join(project_root, "libs")

    if os.path.exists(local_lib_path):
        print(f"📂 [Launcher] 프로젝트 내장 라이브러리 감지: {local_lib_path}")
        dll_paths.add(local_lib_path)
    
    # site-packages 경로 수집
    site_packages_list = site.getsitepackages()
    user_site = site.getusersitepackages()
    if os.path.exists(user_site):
        site_packages_list.append(user_site)
    
    print(f"🔍 [Launcher] 라이브러리 탐색 경로: {len(site_packages_list)}개 소스 + 내장 libs")

    for sp in site_packages_list:
        if not os.path.exists(sp): continue

        # 1. 'nvidia' 폴더 내부
        nvidia_root = os.path.join(sp, "nvidia")
        if os.path.exists(nvidia_root):
            for root, dirs, files in os.walk(nvidia_root):
                if any(f.endswith('.dll') for f in files):
                    dll_paths.add(root)

        # 2. 'torch/lib' 폴더
        torch_lib = os.path.join(sp, "torch", "lib")
        if os.path.exists(torch_lib):
             dll_paths.add(torch_lib)
                        
    return list(dll_paths)

def main():
    print("========================================================")
    print("   MUSE Launcher (Self-Contained Mode v3.0)")
    print("========================================================")

    # 1. 라이브러리 경로 찾기
    nvidia_paths = find_nvidia_dll_paths()
    
    if not nvidia_paths:
        print("⚠️ Warning: NVIDIA 라이브러리 경로를 찾지 못했습니다.")
    else:
        print(f"✅ 로드된 라이브러리 경로 수: {len(nvidia_paths)}개")
        # libs 폴더가 있는지 확인
        has_local_lib = any("libs" in p for p in nvidia_paths)
        if has_local_lib:
            print("   -> 🌟 프로젝트 내부 'libs' 폴더가 우선 적용됩니다.")

    # 2. 환경 변수 PATH 업데이트
    current_path = os.environ.get('PATH', '')
    new_path = os.pathsep.join(nvidia_paths) + os.pathsep + current_path
    
    env = os.environ.copy()
    env['PATH'] = new_path
    
    # [중요] 추가 환경변수 설정
    for p in nvidia_paths:
        if 'cudnn' in p.lower() or 'torch' in p.lower() or 'libs' in p.lower():
            env['CUDNN_PATH'] = p
            env['LD_LIBRARY_PATH'] = p 

    # 3. 메인 프로그램 실행
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(project_root, "src", "main.py")
    
    print("-" * 60)
    print(f"🚀 Launching MUSE: {main_script}")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, main_script], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 실행 중 오류 발생 (Code {e.returncode})")
        print("👉 팁: 프로젝트 폴더 안에 'libs' 폴더를 만들고 'cudnn64_8.dll'을 넣었는지 확인하세요.")
    except KeyboardInterrupt:
        print("\n🛑 종료합니다.")

if __name__ == "__main__":
    main()