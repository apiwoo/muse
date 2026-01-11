# Project MUSE - prepare_onnx_models.py
# Prepares ONNX models for distribution
# Run this on the development PC before creating a distribution package
# (C) 2025 MUSE Corp. All rights reserved.

"""
ONNX Model Preparation Script

This script ensures all required ONNX models are available for distribution.
It converts PyTorch (.pth) models to ONNX format where necessary.

ONNX files are portable across different GPUs, unlike TensorRT engines.
The distribution package includes ONNX files, and TensorRT engines are
built on the user's machine during first run.

Required Models:
1. MODNet (segmentation) - Usually already in ONNX format
2. ViTPose (pose estimation) - Convert from .pth if needed
3. BiSeNet (skin parsing) - Convert from .pth if needed

Usage:
    python tools/prepare_onnx_models.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def check_modnet():
    """Check if MODNet ONNX exists."""
    model_dir = os.path.join(BASE_DIR, "assets", "models", "segmentation")
    onnx_path = os.path.join(model_dir, "modnet.onnx")

    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"[OK] MODNet ONNX: {onnx_path}")
        print(f"     Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"[MISSING] MODNet ONNX not found: {onnx_path}")
        print("     -> Run 'python tools/download_models.py' to download")
        return False


def check_vitpose():
    """Check and prepare ViTPose ONNX."""
    model_dir = os.path.join(BASE_DIR, "assets", "models", "tracking")
    onnx_path = os.path.join(model_dir, "vitpose_huge.onnx")
    pth_path = os.path.join(model_dir, "vitpose_huge_coco_256x192.pth")

    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"[OK] ViTPose ONNX: {onnx_path}")
        print(f"     Size: {size_mb:.1f} MB")
        return True

    if os.path.exists(pth_path):
        print(f"[BUILD] Converting ViTPose PTH to ONNX...")
        print(f"     Source: {pth_path}")
        print(f"     Target: {onnx_path}")

        import subprocess
        try:
            result = subprocess.run([
                sys.executable,
                os.path.join(BASE_DIR, "tools", "trt_converter.py"),
                "--export-worker", pth_path, onnx_path, "huge"
            ], check=True, capture_output=True, text=True)
            print(result.stdout)

            if os.path.exists(onnx_path):
                size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                print(f"[OK] ViTPose ONNX created: {size_mb:.1f} MB")
                return True
            else:
                print("[ERROR] ViTPose ONNX not created")
                return False

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] ViTPose conversion failed: {e}")
            if e.stderr:
                print(e.stderr)
            return False
    else:
        print(f"[MISSING] ViTPose source not found: {pth_path}")
        print("     -> Run 'python tools/download_models.py' to download")
        return False


def check_bisenet():
    """Check and prepare BiSeNet ONNX."""
    model_dir = os.path.join(BASE_DIR, "assets", "models", "parsing")
    onnx_path = os.path.join(model_dir, "bisenet_face_parsing.onnx")
    pth_path = os.path.join(model_dir, "79999_iter.pth")

    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"[OK] BiSeNet ONNX: {onnx_path}")
        print(f"     Size: {size_mb:.1f} MB")
        return True

    if os.path.exists(pth_path):
        print(f"[BUILD] Converting BiSeNet PTH to ONNX...")
        print(f"     Source: {pth_path}")
        print(f"     Target: {onnx_path}")

        import subprocess
        try:
            result = subprocess.run([
                sys.executable,
                os.path.join(BASE_DIR, "tools", "convert_bisenet.py"),
                "--export-only"
            ], check=True, capture_output=True, text=True)
            print(result.stdout)

            if os.path.exists(onnx_path):
                size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                print(f"[OK] BiSeNet ONNX created: {size_mb:.1f} MB")
                return True
            else:
                print("[ERROR] BiSeNet ONNX not created")
                return False

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] BiSeNet conversion failed: {e}")
            if e.stderr:
                print(e.stderr)
            return False
    else:
        print(f"[MISSING] BiSeNet source not found: {pth_path}")
        print("     -> Run 'python tools/download_models.py' to download")
        return False


def main():
    print("=" * 60)
    print("   MUSE ONNX Model Preparation (For Distribution)")
    print("=" * 60)
    print()
    print("This script prepares ONNX models for distribution packages.")
    print("ONNX models are GPU-independent and can be distributed to any user.")
    print("TensorRT engines will be built on the user's machine during first run.")
    print()

    results = []

    print("-" * 60)
    print("[1/3] Checking MODNet Segmentation Model")
    print("-" * 60)
    results.append(("MODNet", check_modnet()))
    print()

    print("-" * 60)
    print("[2/3] Checking ViTPose Pose Estimation Model")
    print("-" * 60)
    results.append(("ViTPose", check_vitpose()))
    print()

    print("-" * 60)
    print("[3/3] Checking BiSeNet Skin Parsing Model")
    print("-" * 60)
    results.append(("BiSeNet", check_bisenet()))
    print()

    # Summary
    print("=" * 60)
    print("   Summary")
    print("=" * 60)

    all_ok = True
    for name, ok in results:
        status = "[OK]" if ok else "[MISSING]"
        print(f"   {status} {name}")
        if not ok:
            all_ok = False

    print()

    if all_ok:
        print("[SUCCESS] All ONNX models are ready!")
        print()
        print("Next steps:")
        print("1. Run 'python tools/build_distribution.py' to create distribution package")
        print("2. The package will include ONNX files (TRT engines excluded)")
        print("3. Users will build TRT engines on first run")
    else:
        print("[WARNING] Some models are missing.")
        print()
        print("To fix:")
        print("1. Run 'python tools/download_models.py' to download missing models")
        print("2. Run this script again")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
