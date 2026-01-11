# Project MUSE - src/utils/cuda_helper.py
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import platform
import glob

def setup_cuda_environment():
    """
    [Core] Resolve Windows DLL Loading Issues.
    Prioritizes 'libs' folder in project root.
    """
    if platform.system() != "Windows":
        return

    # print("[FIX] [CUDA Helper] Setting up library paths...")
    
    site_packages = [p for p in sys.path if 'site-packages' in p]
    
    dll_dirs = set()
    
    # [Custom Fix] Add local 'libs'
    if getattr(sys, 'frozen', False):
        # PyInstaller frozen mode: libs is at executable's parent directory
        project_root = os.path.dirname(sys.executable)
    else:
        # Development mode: calculate from file location
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

    local_lib_path = os.path.join(project_root, "libs")
    
    if os.path.exists(local_lib_path):
        dll_dirs.add(local_lib_path)

    # 1. Search NVIDIA packages
    for sp in site_packages:
        nvidia_path = os.path.join(sp, "nvidia")
        if os.path.exists(nvidia_path):
            for root, dirs, files in os.walk(nvidia_path):
                if os.path.basename(root) in ['bin', 'lib']:
                    if any(f.endswith('.dll') for f in files):
                        dll_dirs.add(root)

        # 2. Torch libs
        torch_lib = os.path.join(sp, "torch", "lib")
        if os.path.exists(torch_lib):
            dll_dirs.add(torch_lib)

    # 3. Add to DLL path
    found_cudnn_8 = False
    
    for directory in dll_dirs:
        try:
            os.add_dll_directory(directory)
            os.environ['PATH'] = directory + os.pathsep + os.environ['PATH']
            
            if glob.glob(os.path.join(directory, "cudnn64_8.dll")):
                found_cudnn_8 = True
                
        except Exception:
            pass

    # 4. Diagnose
    if found_cudnn_8:
        # print("   [OK] Mandatory DLL (cudnn64_8.dll) loaded.")
        pass
    else:
        print("   [WARNING] 'cudnn64_8.dll' not found.")
        print(f"      -> Please put the file in: {local_lib_path}")