import sys
import subprocess
import os
import pkg_resources

# ==============================================================================
# [Project MUSE] Environment Setup Script (v9.2 Dependency Fix)
# ==============================================================================

# Critical Versions
CRITICAL_VERSIONS = {
    "numpy": "1.26.4",
    "onnx": "1.14.0",
    "onnxruntime-gpu": "1.16.0",
    "tensorrt": "10.0.1",
    "tensorrt-cu12": "10.0.1",
    "tensorrt-cu12-bindings": "10.0.1",
    "tensorrt-cu12-libs": "10.0.1",
}

def run_pip(args, description=None):
    if description:
        print(f"   ... {description}")
    
    cmd = [sys.executable, "-m", "pip"] + args
    cmd_str = ' '.join(cmd)
    if len(cmd_str) > 200:
        print(f"   $ {cmd_str[:200]} ... (truncated)")
    else:
        print(f"   $ {cmd_str}")
        
    subprocess.check_call(cmd)

def get_installed_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def ask_user_permission(msg):
    while True:
        response = input(f"\n[WARNING]  {msg} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False

def uninstall_conflicts():
    print("\n[CLEAN] [Step 1] Checking for conflicting packages...")
    
    uninstall_list = [
        "tensorrt", "tensorrt-cu12", "tensorrt-cu12-bindings", "tensorrt-cu12-libs",
        "tensorrt-libs", "onnx", "onnxruntime", "onnxruntime-gpu",
        "numpy" 
    ]
    
    targets = [pkg for pkg in uninstall_list if get_installed_version(pkg)]
    
    if targets:
        print(f"   Found conflicting packages: {', '.join(targets)}")
        
        msg = "To ensure a clean installation, these packages need to be removed and re-installed.\n   Do you want to proceed with cleanup?"
        if not ask_user_permission(msg):
            print("   [SKIP]  Skipping cleanup (Not recommended, but proceeding).")
            return

        try:
            run_pip(["uninstall", "-y"] + targets, description="Removing conflicts")
            print("   [OK] Cleanup complete.")
        except subprocess.CalledProcessError:
            print("   [WARNING] Cleanup skipped or failed.")
    else:
        print("   [OK] No conflicts found. Clean start!")

def install_numpy_base():
    print(f"\n[PKG] [Step 2] Installing Base: Numpy == {CRITICAL_VERSIONS['numpy']}")
    run_pip(["install", f"numpy=={CRITICAL_VERSIONS['numpy']}"])

def install_pytorch_cuda():
    print("\n[FIRE] [Step 3] Verifying PyTorch CUDA 12.1 (cu121)...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   [OK] PyTorch GPU version is active: {torch.__version__}")
            return
    except ImportError:
        pass

    run_pip([
        "install",
        "torch", "torchvision", "torchaudio",
        f"numpy=={CRITICAL_VERSIONS['numpy']}", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], description="Installing PyTorch (cu121)")

def install_core_dependencies():
    print("\n[START] [Step 4] Installing Core & AI Dependencies...")
    
    base_deps = [
        "scipy", "pyyaml", "tqdm",
        "opencv-python<4.11", 
        "mediapipe",      
        "insightface",    
        "moderngl",       
        "moderngl-window",
        "pyvirtualcam",   
        "imgui",          
        "PySide6", 
        "pyqtdarktheme",
        "nvidia-cudnn-cu12",
        "pygrabber",
        "timm",        
        "transformers",
        "decord", # [Added] Required for SAM 2 Video Processing
        "git+https://github.com/facebookresearch/segment-anything-2.git"
    ]

    try:
        import cupy
    except ImportError:
        base_deps.append("cupy-cuda12x")

    final_install_list = base_deps.copy()
    
    for pkg, ver in CRITICAL_VERSIONS.items():
        final_install_list.append(f"{pkg}=={ver}")

    try:
        run_pip(["install"] + final_install_list, description="Installing Main Dependencies with Constraints")
        print("\n   [OK] Core dependencies installed.")
    except subprocess.CalledProcessError:
        print("\n   [ERROR] Package installation failed during Step 4.")
        print("   [TIP] Tip: Check for version conflicts in the output above.")
        sys.exit(1)

def verify_install():
    print("\n[SCAN] [Step 5] Verifying Installation Integrity...")
    all_pass = True
    
    for pkg, expected_ver in CRITICAL_VERSIONS.items():
        installed_ver = get_installed_version(pkg)
        if installed_ver == expected_ver:
            print(f"   [OK] {pkg:<20} : {installed_ver} (Matches)")
        else:
            print(f"   [ERROR] {pkg:<20} : {installed_ver} (Expected: {expected_ver})")
            all_pass = False
            
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        print(f"   {'[OK]' if gpu_ok else '[ERROR]'} PyTorch CUDA       : {'Available' if gpu_ok else 'Not Available'}")
        if not gpu_ok: all_pass = False
    except ImportError:
        print("   [ERROR] PyTorch            : Not Installed")
        all_pass = False

    # Check decord explicitly
    try:
        import decord
        print(f"   [OK] decord             : Installed")
    except ImportError:
        print("   [ERROR] decord             : Not Installed")
        all_pass = False

    return all_pass

def main():
    print("============================================================")
    print("   Project MUSE - Environment Setup (v9.2 Dependency Fix)")
    print("============================================================")
    
    uninstall_conflicts()
    install_numpy_base()
    install_pytorch_cuda()
    install_core_dependencies()

    if verify_install():
        print("\n============================================================")
        print("[DONE] Setup Successfully Completed!")
        print("-> Please run 'tools/download_models.py' next.")
        print("============================================================")
    else:
        print("\n============================================================")
        print("[WARNING] Setup Completed with Warnings.")
        print("-> Some versions do not match the target configuration.")
        print("-> Check the '[ERROR]' marks above.")
        print("============================================================")

if __name__ == "__main__":
    main()