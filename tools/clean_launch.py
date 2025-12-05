# Project MUSE - clean_launch.py
# (C) 2025 MUSE Corp.
# Role: Cleans __pycache__ zombies and launches fresh.

import os
import shutil
import sys
import subprocess

def clean_pycache():
    """
    Walks through project and removes __pycache__ folders.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"[CLEAN] [Cleaner] Cleaning cache... ({root_dir})")
    
    count = 0
    for root, dirs, files in os.walk(root_dir):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                count += 1
            except Exception as e:
                print(f"   [WARNING] Delete failed: {pycache_path} ({e})")
                
    if count > 0:
        print(f"[INFO] Deleted {count} cache folders. Code is fresh.")
    else:
        print("[INFO] Already clean.")

def main():
    print("========================================================")
    print("   MUSE Clean Launcher (Cache Free Mode)")
    print("========================================================")
    
    clean_pycache()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(project_root, "src", "main.py")
    
    print("\n[START] MUSE System Restart (Fresh Start)...")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, main_script], check=True)
    except KeyboardInterrupt:
        print("\n[BYE] Terminated.")
    except Exception as e:
        print(f"\n[ERROR] Execution Error: {e}")

if __name__ == "__main__":
    main()