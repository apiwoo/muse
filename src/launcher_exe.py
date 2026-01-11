# Project MUSE - launcher_exe.py
# PyInstaller entry point for the distribution executable
# This file becomes MUSE.exe when built with PyInstaller
# (C) 2025 MUSE Corp. All rights reserved.

"""
MUSE Executable Launcher

This is the main entry point for the PyInstaller-built executable.
It handles:
1. Logging system initialization
2. DLL path configuration (libs folder)
3. Qt environment setup
4. First-run TensorRT engine building
5. Main application launch

The executable expects the following folder structure:
    PROJECT_MUSE/
        MUSE.exe                (this launcher)
        _internal/              (PyInstaller runtime)
        libs/                   (CUDA/TensorRT DLLs)
        assets/                 (models, shaders, etc.)
        src/                    (Python source code)
        logs/                   (log files - auto created)
"""

import os
import sys
import datetime
import traceback


# ============================================================================
# Logging System
# ============================================================================

class LogRedirector:
    """
    Redirects stdout/stderr to both console and log file.

    In frozen mode (no console), only writes to log file.
    In development mode, writes to both console and log file.
    """

    def __init__(self, log_file, original_stream, is_frozen=False):
        """
        Initialize the log redirector.

        Args:
            log_file: File object to write logs to
            original_stream: Original stdout or stderr stream
            is_frozen: True if running as PyInstaller executable
        """
        self.log_file = log_file
        self.original_stream = original_stream
        self.is_frozen = is_frozen

    def write(self, message):
        """Write message to log file and optionally to console."""
        if message:
            # Always write to log file
            try:
                self.log_file.write(message)
                self.log_file.flush()
            except Exception:
                pass

            # Write to console only in development mode
            if not self.is_frozen and self.original_stream:
                try:
                    self.original_stream.write(message)
                    self.original_stream.flush()
                except Exception:
                    pass

    def flush(self):
        """Flush both streams."""
        try:
            self.log_file.flush()
        except Exception:
            pass

        if not self.is_frozen and self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass

    def fileno(self):
        """Return file descriptor for compatibility."""
        if self.original_stream:
            return self.original_stream.fileno()
        return self.log_file.fileno()


# Global log file path for error dialog
_log_file_path = None


def setup_logging(project_root):
    """
    Initialize the logging system.

    Creates a logs directory and redirects stdout/stderr to a log file.
    Log files are named with timestamp: muse_YYYY-MM-DD_HH-MM-SS.log

    Args:
        project_root: Path to the project root directory

    Returns:
        str: Path to the log file, or None if setup failed
    """
    global _log_file_path

    try:
        # Create logs directory
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Generate log file name with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"muse_{timestamp}.log"
        log_path = os.path.join(logs_dir, log_filename)

        # Open log file
        log_file = open(log_path, 'w', encoding='utf-8', buffering=1)
        _log_file_path = log_path

        # Write log header
        is_frozen = getattr(sys, 'frozen', False)
        mode_str = "Frozen (PyInstaller)" if is_frozen else "Development"

        log_file.write("=" * 80 + "\n")
        log_file.write("PROJECT MUSE Log\n")
        log_file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Mode: {mode_str}\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.flush()

        # Redirect stdout and stderr
        sys.stdout = LogRedirector(log_file, sys.__stdout__, is_frozen)
        sys.stderr = LogRedirector(log_file, sys.__stderr__, is_frozen)

        return log_path

    except Exception as e:
        # If logging setup fails, continue without logging
        print(f"[WARNING] Failed to setup logging: {e}")
        return None


def get_log_timestamp():
    """Get current timestamp for log messages."""
    return datetime.datetime.now().strftime("%H:%M:%S")


def log(tag, message):
    """Print a formatted log message."""
    print(f"[{get_log_timestamp()}] [{tag}] {message}")


# ============================================================================
# Error Dialog
# ============================================================================

def show_error_dialog(title, message, log_path=None):
    """
    Show an error dialog to the user.

    Uses PySide6 QMessageBox if available, otherwise just logs.

    Args:
        title: Dialog title
        message: Error message
        log_path: Path to log file (optional, shown in dialog)
    """
    full_message = message
    if log_path:
        full_message += f"\n\nLog file location:\n{log_path}"

    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        # Check if QApplication exists
        app = QApplication.instance()
        created_app = False

        if app is None:
            app = QApplication(sys.argv)
            created_app = True

        # Create and show message box
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)

        if log_path:
            msg_box.setInformativeText(f"Log file:\n{log_path}")

        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setDefaultButton(QMessageBox.Ok)
        msg_box.exec()

        if created_app:
            app.quit()

    except Exception:
        # If Qt dialog fails, just log the error
        print(f"[ERROR DIALOG] {title}: {message}")
        if log_path:
            print(f"[ERROR DIALOG] Log file: {log_path}")


# ============================================================================
# Environment Setup Functions
# ============================================================================

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

        # Use os.add_dll_directory() for Windows 10+ (required for proper DLL loading)
        try:
            os.add_dll_directory(libs_dir)
            log("DLL", f"Added DLL directory: {libs_dir}")
        except (AttributeError, OSError) as e:
            # os.add_dll_directory not available on older Windows or failed
            log("DLL", f"add_dll_directory not available: {e}")

        log("DLL", f"Added libs path: {libs_dir}")
    else:
        log("WARNING", f"libs folder not found: {libs_dir}")
        log("WARNING", "CUDA/TensorRT DLLs may not load correctly.")


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

    log("PATH", f"Python paths configured: {sys.path[:3]}")


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
        log("OK", "All TensorRT engines found. Starting directly...")
        return True

    # Show what's missing
    log("FIRST RUN", "TensorRT engine build required.")
    missing = builder.get_missing_engines()
    log("INFO", f"Missing engines: {len(missing)}")
    for name, onnx, engine, shape in missing:
        log("INFO", f"  - {name}: {os.path.basename(engine)}")

    # Import Qt and show setup wizard
    try:
        from PySide6.QtWidgets import QApplication
        from ui.setup_wizard import SetupWizardDialog
    except ImportError as e:
        log("ERROR", f"Failed to import PySide6: {e}")
        log("ERROR", "GUI libraries may not be properly installed.")
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
        log("OK", "TensorRT engine build completed successfully!")
        return True
    else:
        log("ERROR", "TensorRT engine build failed.")
        log("ERROR", "Please ensure you have an NVIDIA GPU with proper drivers.")
        return False


def run_main_application():
    """
    Launch the main MUSE application.
    """
    log("START", "Launching PROJECT MUSE...")

    try:
        # Import and run the main application
        from main import main as run_main
        run_main()
    except ImportError as e:
        error_msg = f"Failed to import main module: {e}"
        log("ERROR", error_msg)
        log("DEBUG", f"sys.path = {sys.path[:5]}")
        raise RuntimeError(error_msg)
    except Exception as e:
        log("ERROR", f"Application error: {e}")
        raise


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main entry point for the MUSE executable.
    """
    global _log_file_path
    log_path = None

    try:
        # Step 1: Determine project root (needed for logging setup)
        project_root = get_project_root()

        # Step 2: Setup logging FIRST (before any other output)
        log_path = setup_logging(project_root)

        print("=" * 60)
        print("   PROJECT MUSE Launcher")
        print("=" * 60)
        print()

        log("PATH", f"Project Root: {project_root}")
        frozen_status = "Yes (PyInstaller)" if getattr(sys, 'frozen', False) else "No (Development)"
        log("MODE", f"Frozen: {frozen_status}")
        if log_path:
            log("LOG", f"Log file: {log_path}")
        print()

        # Step 3: Setup DLL paths (must be done before importing torch/tensorrt)
        log("SETUP", "Configuring DLL paths...")
        setup_dll_paths(project_root)

        # Step 4: Setup Qt environment
        log("SETUP", "Configuring Qt environment...")
        setup_qt_environment()

        # Step 5: Setup Python paths
        log("SETUP", "Configuring Python paths...")
        setup_python_paths(project_root)

        print()

        # Step 6: Check first run and build engines if needed
        log("CHECK", "Checking TensorRT engines...")
        if not check_first_run(project_root):
            error_msg = "Setup incomplete. TensorRT engine build failed or was cancelled."
            log("EXIT", error_msg)
            show_error_dialog(
                "PROJECT MUSE - Setup Failed",
                "TensorRT engine build failed or was cancelled.\n\n"
                "Please ensure you have:\n"
                "- NVIDIA GPU with CUDA support\n"
                "- Updated GPU drivers\n"
                "- Sufficient disk space",
                log_path
            )
            sys.exit(1)

        print()

        # Step 7: Run the main application
        run_main_application()

    except Exception as e:
        # Catch any unhandled exception
        error_msg = f"Unexpected error: {e}"
        log("FATAL", error_msg)
        log("FATAL", "Full traceback:")
        traceback.print_exc()

        show_error_dialog(
            "PROJECT MUSE - Error",
            f"An unexpected error occurred:\n\n{str(e)}\n\n"
            "Please check the log file for details.",
            log_path or _log_file_path
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
