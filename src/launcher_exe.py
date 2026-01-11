# Project MUSE - launcher_exe.py
# PyInstaller entry point for the distribution executable
# This file becomes MUSE.exe when built with PyInstaller
# (C) 2025 MUSE Corp. All rights reserved.

"""
MUSE Executable Launcher

This is the main entry point for the PyInstaller-built executable.
It handles:
1. DLL path configuration (libs folder)
2. Qt environment setup
3. First-run TensorRT engine building
4. Main application launch

The executable expects the following folder structure:
    PROJECT_MUSE/
        MUSE.exe                (this launcher)
        _internal/              (PyInstaller runtime)
        libs/                   (CUDA/TensorRT DLLs)
        assets/                 (models, shaders, etc.)
        src/                    (Python source code)
"""

import os
import sys


def get_project_root():
    """
    Get the project root directory.

    Handles both PyInstaller frozen mode and development mode.

    Returns:
        str: Path to the project root directory
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        # sys.executable is the path to MUSE.exe
        return os.path.dirname(sys.executable)
    else:
        # Running in development mode
        # This file is in src/, so parent is project root
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_dll_paths(project_root):
    """
    Add the libs folder to the system PATH for DLL loading.

    This ensures CUDA and TensorRT DLLs can be found at runtime.

    Args:
        project_root: Path to the project root directory
    """
    libs_dir = os.path.join(project_root, "libs")

    if os.path.exists(libs_dir):
        # Add libs to the beginning of PATH
        current_path = os.environ.get('PATH', '')
        if libs_dir not in current_path:
            os.environ['PATH'] = libs_dir + os.pathsep + current_path

        # Also set CUDNN_PATH for compatibility
        os.environ['CUDNN_PATH'] = libs_dir

        print(f"[DLL] Added libs path: {libs_dir}")
    else:
        print(f"[WARNING] libs folder not found: {libs_dir}")
        print("          CUDA/TensorRT DLLs may not load correctly.")


def setup_qt_environment():
    """
    Configure Qt environment variables for proper rendering.
    """
    # Force desktop OpenGL (avoid ANGLE issues on Windows)
    os.environ['QT_OPENGL'] = 'desktop'

    # Disable Qt plugin debug messages
    os.environ['QT_LOGGING_RULES'] = '*.debug=false'


def setup_python_paths(project_root):
    """
    Add source directories to Python path.

    Args:
        project_root: Path to the project root directory
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller frozen mode: src is inside _internal/
        internal_path = os.path.join(project_root, "_internal")
        src_path = os.path.join(internal_path, "src")

        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        if internal_path not in sys.path:
            sys.path.insert(0, internal_path)
    else:
        # Development mode: src is at project_root/src
        src_path = os.path.join(project_root, "src")

        if src_path not in sys.path:
            sys.path.insert(0, src_path)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"[PATH] Python paths: {sys.path[:3]}")


def check_first_run(project_root):
    """
    Check if this is the first run and build TensorRT engines if needed.

    On first run, displays a setup wizard that builds GPU-specific
    TensorRT engines from the included ONNX models.

    Args:
        project_root: Path to the project root directory

    Returns:
        bool: True if ready to proceed, False if setup failed
    """
    from setup.first_run_builder import FirstRunBuilder

    builder = FirstRunBuilder(project_root)

    # Check if all engines already exist
    if builder.check_engines_exist():
        print("[OK] All TensorRT engines found. Starting directly...")
        return True

    # Show what's missing
    print("[FIRST RUN] TensorRT engine build required.")
    missing = builder.get_missing_engines()
    print(f"   Missing engines: {len(missing)}")
    for name, onnx, engine, shape in missing:
        print(f"   - {name}: {os.path.basename(engine)}")

    # Import Qt and show setup wizard
    try:
        from PySide6.QtWidgets import QApplication
        from ui.setup_wizard import SetupWizardDialog
    except ImportError as e:
        print(f"[ERROR] Failed to import PySide6: {e}")
        print("        GUI libraries may not be properly installed.")
        return False

    # Check if QApplication already exists
    app = QApplication.instance()
    created_app = False

    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    # Show the setup wizard
    dialog = SetupWizardDialog(builder)
    dialog.exec()

    success = dialog.build_success

    # Clean up if we created the app
    if created_app:
        app.quit()
        del app

    if success:
        print("[OK] TensorRT engine build completed successfully!")
        return True
    else:
        print("[ERROR] TensorRT engine build failed.")
        print("        Please ensure you have an NVIDIA GPU with proper drivers.")
        return False


def run_main_application():
    """
    Launch the main MUSE application.
    """
    print("[START] Launching PROJECT MUSE...")

    try:
        # Import and run the main application
        from main import main as run_main
        run_main()
    except ImportError as e:
        print(f"[ERROR] Failed to import main module: {e}")
        print(f"[DEBUG] sys.path = {sys.path[:5]}")
        print()
        print("Press Enter to close...")
        try:
            input()
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Application error: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Press Enter to close...")
        try:
            input()
        except Exception:
            pass
        sys.exit(1)


def main():
    """
    Main entry point for the MUSE executable.
    """
    print("=" * 60)
    print("   PROJECT MUSE Launcher")
    print("=" * 60)
    print()

    # Step 1: Determine project root
    project_root = get_project_root()
    print(f"[PATH] Project Root: {project_root}")

    frozen_status = "Yes (PyInstaller)" if getattr(sys, 'frozen', False) else "No (Development)"
    print(f"[MODE] Frozen: {frozen_status}")
    print()

    # Step 2: Setup DLL paths (must be done before importing torch/tensorrt)
    print("[SETUP] Configuring DLL paths...")
    setup_dll_paths(project_root)

    # Step 3: Setup Qt environment
    print("[SETUP] Configuring Qt environment...")
    setup_qt_environment()

    # Step 4: Setup Python paths
    print("[SETUP] Configuring Python paths...")
    setup_python_paths(project_root)

    print()

    # Step 5: Check first run and build engines if needed
    print("[CHECK] Checking TensorRT engines...")
    if not check_first_run(project_root):
        print()
        print("[EXIT] Setup incomplete. Exiting...")
        print()
        print("Press Enter to close...")
        try:
            input()
        except Exception:
            pass
        sys.exit(1)

    print()

    # Step 6: Run the main application
    run_main_application()


if __name__ == "__main__":
    main()
