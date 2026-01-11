# Project MUSE - run_muse.py
# Main launcher with first-run TensorRT engine build support
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import glob
import subprocess
import site

# Add src directory to path for imports
_current_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_current_file))
sys.path.insert(0, os.path.join(_project_root, "src"))

def find_nvidia_dll_paths():
    """
    Search for nvidia packages in site-packages and local 'libs'.
    """
    dll_paths = set()
    
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    local_lib_path = os.path.join(project_root, "libs")

    if os.path.exists(local_lib_path):
        print(f"[DIR] [Launcher] Local libs detected: {local_lib_path}")
        dll_paths.add(local_lib_path)
    
    site_packages_list = site.getsitepackages()
    user_site = site.getusersitepackages()
    if os.path.exists(user_site):
        site_packages_list.append(user_site)
    
    print(f"[SCAN] [Launcher] Searching paths: {len(site_packages_list)} sources + local libs")

    for sp in site_packages_list:
        if not os.path.exists(sp): continue

        nvidia_root = os.path.join(sp, "nvidia")
        if os.path.exists(nvidia_root):
            for root, dirs, files in os.walk(nvidia_root):
                if any(f.endswith('.dll') for f in files):
                    dll_paths.add(root)

        torch_lib = os.path.join(sp, "torch", "lib")
        if os.path.exists(torch_lib):
             dll_paths.add(torch_lib)
                        
    return list(dll_paths)

def main():
    print("========================================================")
    print("   MUSE Launcher (Self-Contained Mode v3.0)")
    print("========================================================")

    nvidia_paths = find_nvidia_dll_paths()
    
    if not nvidia_paths:
        print("[WARNING] Warning: NVIDIA library paths not found.")
    else:
        print(f"[OK] Loaded library paths: {len(nvidia_paths)}")
        has_local_lib = any("libs" in p for p in nvidia_paths)
        if has_local_lib:
            print("   -> [STAR] Project local 'libs' will be prioritized.")

    current_path = os.environ.get('PATH', '')
    new_path = os.pathsep.join(nvidia_paths) + os.pathsep + current_path
    
    env = os.environ.copy()
    env['PATH'] = new_path
    
    for p in nvidia_paths:
        if 'cudnn' in p.lower() or 'torch' in p.lower() or 'libs' in p.lower():
            env['CUDNN_PATH'] = p
            env['LD_LIBRARY_PATH'] = p

    # OpenGL 설정 - Qt가 D3D11 대신 OpenGL을 사용하도록 강제
    env['QT_OPENGL'] = 'desktop'

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(project_root, "src", "main.py")

    # === First Run Check: TensorRT Engine Build ===
    try:
        from setup.first_run_builder import FirstRunBuilder

        builder = FirstRunBuilder(project_root)

        if not builder.check_engines_exist():
            print("-" * 60)
            print("[FIRST RUN] TensorRT engine build required.")
            print("[FIRST RUN] Launching setup wizard...")
            print("-" * 60)

            # Import PySide6 for UI
            from PySide6.QtWidgets import QApplication
            from ui.setup_wizard import SetupWizardDialog

            # Create Qt application
            app = QApplication(sys.argv)

            # Show setup wizard
            dialog = SetupWizardDialog(builder)
            dialog.exec()

            if not dialog.build_success:
                print("[ERROR] Engine build failed. Exiting.")
                sys.exit(1)

            # Clean up Qt application
            app.quit()
            del app

            print("[FIRST RUN] Engine build complete. Starting main application...")
    except ImportError as e:
        # If setup module not available, skip first-run check
        print(f"[WARNING] First-run check skipped: {e}")
    except Exception as e:
        print(f"[WARNING] First-run check error: {e}")
        # Continue anyway - main app might still work

    # === Launch Main Application ===
    print("-" * 60)
    print(f"[START] Launching MUSE: {main_script}")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, main_script], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Execution failed (Code {e.returncode})")
        print("[TIP] Tip: Ensure 'libs' folder with 'cudnn64_8.dll' exists in project root.")
    except KeyboardInterrupt:
        print("\n[STOP] Terminated.")

if __name__ == "__main__":
    main()