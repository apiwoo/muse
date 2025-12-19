# Project MUSE - env_doctor.py
# Unified Environment Repair Tool
# Consolidates: patch_dll.py, force_patch_ort.py, fix_env.py
# (C) 2025 MUSE Corp. All rights reserved.

"""
MUSE Environment Doctor - Unified Environment Repair Tool

This tool consolidates all environment repair functions:
1. DLL Patching (from patch_dll.py) - Copy CUDA DLLs to project root
2. ONNXRuntime Force Patching (from force_patch_ort.py) - Inject DLLs into ORT capi
3. Environment Fixing (from fix_env.py) - Fix TensorRT/ONNX version conflicts

Usage:
    python env_doctor.py --diagnose     # Check environment status
    python env_doctor.py --fix-dll      # Fix DLL loading issues
    python env_doctor.py --fix-ort      # Fix ONNXRuntime issues
    python env_doctor.py --fix-trt      # Fix TensorRT version conflicts
    python env_doctor.py --fix-all      # Run all fixes
"""

import os
import sys
import shutil
import glob
import site
import subprocess
import argparse

try:
    import pkg_resources
    HAS_PKG_RESOURCES = True
except ImportError:
    HAS_PKG_RESOURCES = False


class EnvDoctor:
    """Unified environment repair tool for Project MUSE."""

    def __init__(self, project_root=None):
        """
        Initialize EnvDoctor.

        Args:
            project_root: Path to project root. Auto-detected if None.
        """
        if project_root is None:
            # Auto-detect project root (2 levels up from this file)
            self.project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        else:
            self.project_root = project_root

        self.site_packages = self._get_site_packages()

    def _get_site_packages(self):
        """Get the primary site-packages directory."""
        try:
            return site.getsitepackages()[0]
        except (AttributeError, IndexError):
            # Fallback for virtual environments
            for p in sys.path:
                if 'site-packages' in p and os.path.isdir(p):
                    return p
            return None

    def _get_installed_version(self, package_name):
        """Get installed version of a package."""
        if not HAS_PKG_RESOURCES:
            return "Unknown"
        try:
            return pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            return "Not Installed"

    def _run_cmd(self, cmd, verbose=True):
        """Run a shell command."""
        if verbose:
            print(f"   $ {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
            return True
        except subprocess.CalledProcessError:
            if verbose:
                print("   Warning: Command failed, continuing...")
            return False

    # =========================================================================
    # Diagnosis Functions
    # =========================================================================

    def diagnose(self):
        """Run full environment diagnosis."""
        print("=" * 60)
        print("   MUSE Environment Doctor - Diagnosis")
        print("=" * 60)

        print(f"\nProject Root: {self.project_root}")
        print(f"Site-Packages: {self.site_packages}")

        # Check Python version
        print(f"\nPython Version: {sys.version}")

        # Check key packages
        print("\n[Package Versions]")
        packages = [
            "cupy-cuda12x", "onnxruntime-gpu", "onnx",
            "tensorrt", "tensorrt-cu12", "tensorrt-cu12-bindings", "tensorrt-cu12-libs",
            "mediapipe", "opencv-python"
        ]
        for pkg in packages:
            ver = self._get_installed_version(pkg)
            status = "OK" if ver != "Not Installed" else "MISSING"
            print(f"   {pkg}: {ver} [{status}]")

        # Check NVIDIA packages
        print("\n[NVIDIA DLL Packages]")
        nvidia_dir = os.path.join(self.site_packages, "nvidia") if self.site_packages else None
        if nvidia_dir and os.path.exists(nvidia_dir):
            nvidia_pkgs = os.listdir(nvidia_dir)
            for pkg in nvidia_pkgs[:10]:  # Show first 10
                print(f"   - {pkg}")
            if len(nvidia_pkgs) > 10:
                print(f"   ... and {len(nvidia_pkgs) - 10} more")
        else:
            print("   NVIDIA packages not found!")

        # Check ONNXRuntime capi
        print("\n[ONNXRuntime Status]")
        ort_dirs = self._find_ort_capi_dirs()
        if ort_dirs:
            for d in ort_dirs:
                dll_count = len(glob.glob(os.path.join(d, "*.dll")))
                print(f"   {d}: {dll_count} DLLs")
        else:
            print("   ONNXRuntime not found!")

        print("\n" + "=" * 60)

    def _find_nvidia_packages(self):
        """Find nvidia package paths in site-packages."""
        nvidia_paths = []
        for p in sys.path:
            if 'site-packages' in p:
                nv_path = os.path.join(p, "nvidia")
                if os.path.exists(nv_path):
                    nvidia_paths.append(nv_path)
        return nvidia_paths

    def _find_ort_capi_dirs(self):
        """Find ONNXRuntime capi directories."""
        if not self.site_packages:
            return []

        ort_dirs = []
        candidates = glob.glob(os.path.join(self.site_packages, "onnxruntime*"))
        for d in candidates:
            if os.path.isdir(d):
                capi_path = os.path.join(d, "capi")
                if os.path.exists(capi_path):
                    ort_dirs.append(capi_path)
        return ort_dirs

    # =========================================================================
    # Fix Functions
    # =========================================================================

    def fix_dll(self):
        """
        Fix DLL loading issues by copying CUDA DLLs to project root.
        (Equivalent to patch_dll.py)
        """
        print("=" * 60)
        print("   MUSE DLL Patcher (Fix Error 126)")
        print("=" * 60)

        print(f"\nTarget Directory: {self.project_root}")

        nvidia_roots = self._find_nvidia_packages()
        if not nvidia_roots:
            print("ERROR: NVIDIA packages not found!")
            print("   Run: pip install nvidia-cudnn-cu12 nvidia-cublas-cu12")
            return False

        dll_patterns = [
            "cudnn*/bin/*.dll",
            "cublas*/bin/*.dll",
            "cufft*/bin/*.dll",
            "curand*/bin/*.dll",
            "cuda_runtime*/bin/*.dll"
        ]

        count = 0
        for nv_root in nvidia_roots:
            print(f"\nScanning: {nv_root}")
            for pattern in dll_patterns:
                search_path = os.path.join(nv_root, pattern)
                found_dlls = glob.glob(search_path)

                for dll_path in found_dlls:
                    filename = os.path.basename(dll_path)
                    target_path = os.path.join(self.project_root, filename)

                    if not os.path.exists(target_path):
                        try:
                            shutil.copy2(dll_path, target_path)
                            print(f"   Copied: {filename}")
                            count += 1
                        except Exception as e:
                            print(f"   Failed: {filename} ({e})")

        if count == 0:
            print("\nNo new files copied (already exist or not found)")
        else:
            print(f"\nCopied {count} DLL files to project root")

        return True

    def fix_ort(self):
        """
        Fix ONNXRuntime by injecting NVIDIA DLLs directly into capi folder.
        (Equivalent to force_patch_ort.py)
        """
        print("=" * 60)
        print("   MUSE ONNXRuntime Force Patcher")
        print("=" * 60)

        if not self.site_packages:
            print("ERROR: Cannot find site-packages directory!")
            return False

        target_dirs = self._find_ort_capi_dirs()
        if not target_dirs:
            print("ERROR: ONNXRuntime not found!")
            return False

        print(f"\nTarget Directories ({len(target_dirs)} found):")
        for d in target_dirs:
            print(f"   - {d}")

        nvidia_dir = os.path.join(self.site_packages, "nvidia")
        if not os.path.exists(nvidia_dir):
            print("ERROR: NVIDIA packages not found!")
            return False

        total_count = 0
        print("\nInjecting DLLs into ONNXRuntime...")

        for root, dirs, files in os.walk(nvidia_dir):
            for file in files:
                if file.endswith(".dll"):
                    src_path = os.path.join(root, file)

                    for target_dir in target_dirs:
                        dst_path = os.path.join(target_dir, file)
                        try:
                            shutil.copy2(src_path, dst_path)
                            total_count += 1
                        except Exception as e:
                            print(f"   Failed: {file} -> {target_dir} ({e})")

        if total_count > 0:
            print(f"\nSuccess! Copied {total_count} files")
        else:
            print("\nNo DLLs found to copy")

        return True

    def fix_trt(self, interactive=True):
        """
        Fix TensorRT version conflicts by reinstalling with pinned versions.
        (Equivalent to fix_env.py)
        """
        print("=" * 60)
        print("   MUSE TensorRT Version Fixer")
        print("=" * 60)

        packages = [
            "tensorrt", "tensorrt-cu12", "tensorrt-cu12-bindings", "tensorrt-cu12-libs",
            "onnx", "onnxruntime-gpu"
        ]

        print("\n[Current Versions]")
        for pkg in packages:
            ver = self._get_installed_version(pkg)
            print(f"   {pkg}: {ver}")

        if interactive:
            print("\nWARNING: This will uninstall and reinstall TensorRT/ONNX packages.")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return False

        # Uninstall
        print("\n[Uninstalling existing packages...]")
        uninstall_list = [
            "tensorrt", "tensorrt-cu12", "tensorrt-cu12-bindings", "tensorrt-cu12-libs",
            "tensorrt-libs", "onnx", "onnxruntime", "onnxruntime-gpu"
        ]
        self._run_cmd([sys.executable, "-m", "pip", "uninstall", "-y"] + uninstall_list)

        # Reinstall with pinned versions
        print("\n[Installing pinned versions...]")
        install_cmds = [
            ["onnx==1.14.0"],
            ["onnxruntime-gpu==1.16.0"],
            ["tensorrt==10.0.1", "tensorrt-cu12==10.0.1",
             "tensorrt-cu12-bindings==10.0.1", "tensorrt-cu12-libs==10.0.1"]
        ]

        for pkgs in install_cmds:
            self._run_cmd([sys.executable, "-m", "pip", "install"] + pkgs)

        print("\n[Final Versions]")
        for pkg in packages:
            ver = self._get_installed_version(pkg)
            print(f"   {pkg}: {ver}")

        print("\nDone!")
        return True

    def fix_all(self):
        """Run all fixes in sequence."""
        print("Running all fixes...\n")
        self.fix_dll()
        print()
        self.fix_ort()
        print()
        self.fix_trt(interactive=False)
        print("\nAll fixes completed!")


def main():
    parser = argparse.ArgumentParser(
        description="MUSE Environment Doctor - Unified Environment Repair Tool"
    )
    parser.add_argument("--diagnose", action="store_true",
                       help="Run environment diagnosis")
    parser.add_argument("--fix-dll", action="store_true",
                       help="Fix DLL loading issues (Error 126)")
    parser.add_argument("--fix-ort", action="store_true",
                       help="Fix ONNXRuntime issues")
    parser.add_argument("--fix-trt", action="store_true",
                       help="Fix TensorRT version conflicts")
    parser.add_argument("--fix-all", action="store_true",
                       help="Run all fixes")

    args = parser.parse_args()

    doctor = EnvDoctor()

    if args.diagnose:
        doctor.diagnose()
    elif args.fix_dll:
        doctor.fix_dll()
    elif args.fix_ort:
        doctor.fix_ort()
    elif args.fix_trt:
        doctor.fix_trt()
    elif args.fix_all:
        doctor.fix_all()
    else:
        # Default: show diagnosis
        doctor.diagnose()
        print("\nUsage: python env_doctor.py [--diagnose|--fix-dll|--fix-ort|--fix-trt|--fix-all]")


if __name__ == "__main__":
    main()
