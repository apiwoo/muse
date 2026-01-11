# Project MUSE - build_distribution.py
# Creates PyInstaller-based distribution package for deployment
# (C) 2025 MUSE Corp. All rights reserved.

"""
Distribution Package Builder (PyInstaller Version)

Creates a deployable package of PROJECT MUSE using PyInstaller.
The package includes:
- MUSE.exe (PyInstaller-generated executable)
- _internal/ (Python runtime and packages)
- libs/ (CUDA/TensorRT DLLs)
- assets/ (ONNX models, shaders, configs)
- src/ (Python source code for dynamic imports)

Workflow:
1. Check ONNX models exist (run prepare_onnx_models.py first if missing)
2. Check libs folder exists
3. Run PyInstaller with muse.spec
4. Copy libs/, assets/ to dist folder
5. Create README.txt
6. Optionally create ZIP archive

Usage:
    python tools/build_distribution.py
"""

import os
import sys
import shutil
import subprocess
import zipfile
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def print_header(title):
    """Print a formatted section header."""
    print()
    print("=" * 60)
    print(f"   {title}")
    print("=" * 60)


def print_step(step_num, total, description):
    """Print a step indicator."""
    print()
    print(f"[{step_num}/{total}] {description}")
    print("-" * 40)


def check_onnx_models():
    """
    Verify all required ONNX models exist.

    Returns:
        tuple: (success: bool, details: list of tuples (name, path, exists, size_mb))
    """
    model_dir = os.path.join(BASE_DIR, "assets", "models")

    required = [
        ("MODNet", os.path.join(model_dir, "segmentation", "modnet.onnx")),
        ("ViTPose", os.path.join(model_dir, "tracking", "vitpose_huge.onnx")),
        ("BiSeNet", os.path.join(model_dir, "parsing", "bisenet_face_parsing.onnx")),
    ]

    results = []
    all_exist = True

    for name, path in required:
        exists = os.path.exists(path)
        size_mb = os.path.getsize(path) / (1024 * 1024) if exists else 0

        results.append((name, path, exists, size_mb))

        if exists:
            print(f"   [OK] {name}: {size_mb:.1f} MB")
        else:
            print(f"   [MISSING] {name}")
            print(f"             {path}")
            all_exist = False

    return all_exist, results


def check_libs_folder():
    """
    Verify libs folder with DLLs exists.

    Returns:
        tuple: (success: bool, dll_count: int)
    """
    libs_dir = os.path.join(BASE_DIR, "libs")

    if not os.path.exists(libs_dir):
        print(f"   [MISSING] libs folder not found")
        print(f"             {libs_dir}")
        return False, 0

    # Count DLL files
    dll_files = [f for f in os.listdir(libs_dir) if f.lower().endswith('.dll')]
    dll_count = len(dll_files)

    print(f"   [OK] libs folder found: {dll_count} DLL files")

    # Check critical DLLs
    critical_patterns = ['cudnn', 'nvinfer', 'cublas', 'cudart', 'nvrtc']
    missing_critical = []

    for pattern in critical_patterns:
        found = any(pattern in f.lower() for f in dll_files)
        if not found:
            missing_critical.append(pattern)

    if missing_critical:
        print(f"   [WARNING] Some critical DLLs may be missing:")
        for pattern in missing_critical:
            print(f"             - {pattern}*.dll")

    return True, dll_count


def check_spec_file():
    """
    Verify muse.spec exists.

    Returns:
        bool: True if spec file exists
    """
    spec_path = os.path.join(BASE_DIR, "muse.spec")

    if os.path.exists(spec_path):
        print(f"   [OK] muse.spec found")
        return True
    else:
        print(f"   [MISSING] muse.spec not found")
        print(f"             {spec_path}")
        return False


def run_pyinstaller():
    """
    Run PyInstaller with muse.spec.

    Returns:
        bool: True if build successful
    """
    spec_path = os.path.join(BASE_DIR, "muse.spec")

    print("   Running PyInstaller... (this may take several minutes)")
    print()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "PyInstaller", spec_path, "--noconfirm"],
            cwd=BASE_DIR,
            capture_output=False,  # Show output in real-time
        )

        if result.returncode == 0:
            print()
            print("   [OK] PyInstaller build completed successfully")
            return True
        else:
            print()
            print(f"   [ERROR] PyInstaller failed with code {result.returncode}")
            return False

    except FileNotFoundError:
        print("   [ERROR] PyInstaller not found. Install with: pip install pyinstaller")
        return False
    except Exception as e:
        print(f"   [ERROR] PyInstaller failed: {e}")
        return False


def copy_libs_folder(dist_dir):
    """
    Copy libs folder to distribution.

    Args:
        dist_dir: Distribution output directory

    Returns:
        bool: True if successful
    """
    libs_src = os.path.join(BASE_DIR, "libs")
    libs_dst = os.path.join(dist_dir, "libs")

    if not os.path.exists(libs_src):
        print("   [ERROR] libs folder not found")
        return False

    print(f"   Copying libs folder...")

    try:
        if os.path.exists(libs_dst):
            shutil.rmtree(libs_dst)

        shutil.copytree(libs_src, libs_dst)

        dll_count = len([f for f in os.listdir(libs_dst) if f.lower().endswith('.dll')])
        print(f"   [OK] Copied {dll_count} DLL files")
        return True

    except Exception as e:
        print(f"   [ERROR] Failed to copy libs: {e}")
        return False


def copy_assets_folder(dist_dir):
    """
    Copy assets folder to distribution, excluding TensorRT engines.

    Args:
        dist_dir: Distribution output directory

    Returns:
        bool: True if successful
    """
    assets_src = os.path.join(BASE_DIR, "assets")
    assets_dst = os.path.join(dist_dir, "assets")

    if not os.path.exists(assets_src):
        print("   [WARNING] assets folder not found - skipping")
        return True

    print(f"   Copying assets folder...")

    # Extensions to exclude (GPU-specific files)
    exclude_extensions = {'.engine', '.pth', '.ckpt', '.pt'}

    try:
        if os.path.exists(assets_dst):
            shutil.rmtree(assets_dst)

        def ignore_func(directory, files):
            ignored = []
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in exclude_extensions:
                    ignored.append(f)
            return ignored

        shutil.copytree(assets_src, assets_dst, ignore=ignore_func)

        # Count ONNX files copied
        onnx_count = 0
        for root, dirs, files in os.walk(assets_dst):
            onnx_count += len([f for f in files if f.endswith('.onnx')])

        print(f"   [OK] Copied assets folder ({onnx_count} ONNX models)")
        print(f"        (Excluded .engine, .pth, .ckpt files)")
        return True

    except Exception as e:
        print(f"   [ERROR] Failed to copy assets: {e}")
        return False


def create_user_directories(dist_dir):
    """
    Create empty user data directories.

    Args:
        dist_dir: Distribution output directory
    """
    user_dirs = [
        os.path.join(dist_dir, "recorded_data", "personal_data"),
        os.path.join(dist_dir, "recorded_data", "sessions"),
    ]

    for d in user_dirs:
        os.makedirs(d, exist_ok=True)

    print("   [OK] Created user data directories")


def create_readme(dist_dir):
    """
    Create README.txt file.

    Args:
        dist_dir: Distribution output directory
    """
    readme_content = """PROJECT MUSE - User Guide
========================

[System Requirements]
- Windows 10/11 64-bit
- NVIDIA Graphics Card (RTX 20 series or newer recommended)
- NVIDIA Driver installed (version 525+ recommended)
- 8GB+ RAM
- 4GB+ VRAM

[Installation]
1. Extract this folder to your desired location
2. Double-click MUSE.exe to run

[First Run]
On first launch, the program will optimize AI models for your GPU.
This process takes approximately 5-10 minutes and only runs once.
Please do not close the window during this process.

[Folder Structure]
- MUSE.exe        : Main executable
- _internal/      : Python runtime (do not modify)
- libs/           : GPU libraries (CUDA/TensorRT)
- assets/         : AI models and resources
- recorded_data/  : Your saved data

[Troubleshooting]

"NVIDIA GPU not found" error:
  -> Update NVIDIA driver to the latest version
  -> https://www.nvidia.com/drivers

Program does not start:
  -> Ensure 'libs' folder exists
  -> Install Visual C++ Redistributable 2019+
  -> https://aka.ms/vs/17/release/vc_redist.x64.exe

TensorRT engine build fails:
  -> Ensure you have at least 4GB free VRAM
  -> Close other GPU-intensive applications
  -> Try running as administrator

[Support]
For issues and feedback, please contact the development team.

[License]
(C) 2025 MUSE Corp. All rights reserved.
"""

    readme_path = os.path.join(dist_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("   [OK] Created README.txt")


def calculate_folder_size(folder):
    """Calculate total size of a folder in bytes."""
    total = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def create_zip_archive(dist_dir):
    """
    Create ZIP archive of the distribution.

    Args:
        dist_dir: Distribution output directory

    Returns:
        str: Path to created ZIP file
    """
    zip_path = dist_dir + ".zip"

    print(f"   Creating ZIP archive...")
    print(f"   Target: {os.path.basename(zip_path)}")

    # Directories to exclude from ZIP
    exclude_dirs = {'__pycache__', '.git', '.idea', '.vscode'}

    file_count = 0

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(dist_dir):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                file_path = os.path.join(root, file)
                # Archive name relative to parent of dist_dir
                arc_name = os.path.relpath(file_path, os.path.dirname(dist_dir))
                zf.write(file_path, arc_name)
                file_count += 1

    zip_size = os.path.getsize(zip_path)
    zip_size_gb = zip_size / (1024 * 1024 * 1024)
    zip_size_mb = zip_size / (1024 * 1024)

    if zip_size_gb >= 1:
        size_str = f"{zip_size_gb:.2f} GB"
    else:
        size_str = f"{zip_size_mb:.1f} MB"

    print(f"   [OK] Created ZIP archive")
    print(f"        Files: {file_count}")
    print(f"        Size: {size_str}")

    return zip_path


def main():
    print_header("PROJECT MUSE Distribution Builder (PyInstaller)")

    total_steps = 7

    # Step 1: Check ONNX models
    print_step(1, total_steps, "Checking ONNX models...")
    onnx_ok, _ = check_onnx_models()

    if not onnx_ok:
        print()
        print("[ERROR] Required ONNX models are missing.")
        print("        Run 'python tools/prepare_onnx_models.py' first.")
        sys.exit(1)

    # Step 2: Check libs folder
    print_step(2, total_steps, "Checking libs folder...")
    libs_ok, dll_count = check_libs_folder()

    if not libs_ok:
        print()
        print("[ERROR] libs folder is missing.")
        print("        Ensure CUDA/TensorRT DLLs are in the libs/ folder.")
        sys.exit(1)

    # Step 3: Check spec file
    print_step(3, total_steps, "Checking PyInstaller spec file...")
    spec_ok = check_spec_file()

    if not spec_ok:
        print()
        print("[ERROR] muse.spec file is missing.")
        print("        This file should be in the project root.")
        sys.exit(1)

    # Step 4: Run PyInstaller
    print_step(4, total_steps, "Running PyInstaller...")
    pyinstaller_ok = run_pyinstaller()

    if not pyinstaller_ok:
        print()
        print("[ERROR] PyInstaller build failed.")
        print("        Check the error messages above.")
        sys.exit(1)

    # Distribution output directory
    dist_dir = os.path.join(BASE_DIR, "dist", "PROJECT_MUSE")

    if not os.path.exists(dist_dir):
        print()
        print(f"[ERROR] Distribution folder not created: {dist_dir}")
        sys.exit(1)

    # Step 5: Copy additional files
    print_step(5, total_steps, "Copying additional files...")

    libs_copied = copy_libs_folder(dist_dir)
    assets_copied = copy_assets_folder(dist_dir)
    create_user_directories(dist_dir)
    create_readme(dist_dir)

    if not libs_copied:
        print()
        print("[WARNING] libs folder copy failed. Distribution may not work.")

    # Step 6: Calculate size
    print_step(6, total_steps, "Calculating distribution size...")

    total_size = calculate_folder_size(dist_dir)
    size_gb = total_size / (1024 * 1024 * 1024)
    size_mb = total_size / (1024 * 1024)

    if size_gb >= 1:
        size_str = f"{size_gb:.2f} GB"
    else:
        size_str = f"{size_mb:.1f} MB"

    print(f"   Distribution size: {size_str}")

    # Step 7: Create ZIP (optional)
    print_step(7, total_steps, "Create ZIP archive?")

    try:
        response = input("   Create ZIP archive for distribution? [y/N]: ").strip().lower()
    except EOFError:
        response = 'n'

    zip_path = None
    if response == 'y':
        zip_path = create_zip_archive(dist_dir)

    # Summary
    print_header("Build Complete!")

    print(f"   Distribution folder:")
    print(f"   {dist_dir}")
    print()
    print(f"   Size: {size_str}")

    if zip_path:
        zip_size = os.path.getsize(zip_path) / (1024 * 1024 * 1024)
        print()
        print(f"   ZIP archive:")
        print(f"   {zip_path}")
        print(f"   ZIP size: {zip_size:.2f} GB")

    print()
    print("   Next steps:")
    print("   1. Test MUSE.exe on your development machine")
    print("   2. Test on a clean machine without Python installed")
    print("   3. Upload ZIP to Google Drive for distribution")
    print()


if __name__ == "__main__":
    main()
