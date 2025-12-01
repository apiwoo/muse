# Project MUSE - clean_launch.py
# (C) 2025 MUSE Corp.
# 역할: __pycache__ 좀비 파일을 삭제하고 깨끗하게 실행합니다.

import os
import shutil
import sys
import subprocess

def clean_pycache():
    """
    프로젝트 전체를 돌면서 __pycache__ 폴더를 찾아 삭제합니다.
    """
    # tools 폴더의 상위(프로젝트 루트)를 기준으로 함
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"🧹 [Cleaner] 캐시 파일 청소 중... ({root_dir})")
    
    count = 0
    for root, dirs, files in os.walk(root_dir):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path) # 폴더 강제 삭제
                # print(f"   -> Deleted: {pycache_path}")
                count += 1
            except Exception as e:
                print(f"   ⚠️ 삭제 실패: {pycache_path} ({e})")
                
    if count > 0:
        print(f"✨ 총 {count}개의 캐시 폴더를 삭제했습니다. 이제 코드가 확실히 반영됩니다.")
    else:
        print("✨ 이미 깨끗합니다.")

def main():
    print("========================================================")
    print("   MUSE Clean Launcher (Cache Free Mode)")
    print("========================================================")
    
    # 1. 청소 수행
    clean_pycache()
    
    # 2. 메인 프로그램 경로 설정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(project_root, "src", "main.py")
    
    # 3. 깨끗한 상태로 실행
    print("\n🚀 MUSE 시스템 재시작 (Fresh Start)...")
    print("-" * 60)
    
    try:
        # 현재 파이썬 실행파일로 main.py 구동
        subprocess.run([sys.executable, main_script], check=True)
    except KeyboardInterrupt:
        print("\n👋 종료합니다.")
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()