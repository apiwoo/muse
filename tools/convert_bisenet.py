# Project MUSE - convert_bisenet.py
# BiSeNet V2 Full Conversion Pipeline: PyTorch → ONNX → TensorRT
# Converts face parsing model for SkinParser
# (C) 2025 MUSE Corp. All rights reserved.

"""
BiSeNet V2 Face Parsing Model Converter

Full pipeline for converting BiSeNet V2 from PyTorch checkpoint to TensorRT engine.

Model Source:
- CelebAMask-HQ pretrained BiSeNet V2
- GitHub: https://github.com/zllrunning/face-parsing.PyTorch
- Input: 512x512 RGB image
- Output: 19-class segmentation map (class 1 = skin)

Usage:
    # Full pipeline (PyTorch → ONNX → TensorRT):
    python tools/convert_bisenet.py

    # Only ONNX export:
    python tools/convert_bisenet.py --export-only

    # Only TensorRT build (from existing ONNX):
    python tools/convert_bisenet.py --build-only
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Default paths
MODEL_DIR = os.path.join(BASE_DIR, "assets", "models", "parsing")
PTH_PATH = os.path.join(MODEL_DIR, "79999_iter.pth")  # Original face-parsing.PyTorch checkpoint
ONNX_PATH = os.path.join(MODEL_DIR, "bisenet_face_parsing.onnx")
ENGINE_PATH = os.path.join(MODEL_DIR, "bisenet_v2_fp16.engine")


# ==============================================================================
# BiSeNet Model Architecture (face-parsing.PyTorch compatible)
# Reference: https://github.com/zllrunning/face-parsing.PyTorch
# ==============================================================================
def get_bisenet_model():
    """
    Build BiSeNet model architecture matching face-parsing.PyTorch exactly.

    Architecture:
    - ResNet18 backbone: feat8(128ch), feat16(256ch), feat32(512ch)
    - ARM16: 256→128, ARM32: 512→128
    - ContextPath: combines ARM outputs
    - Spatial: uses feat8 (128ch) directly
    - FFM: concat(128, 128)=256 → 256
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvBNReLU(nn.Module):
        def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks,
                                  stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(out_chan)
            # Initialize weights
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = F.relu(x)
            return x

    class BiSeNetOutput(nn.Module):
        def __init__(self, in_chan, mid_chan, n_classes):
            super().__init__()
            self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
            self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
            nn.init.kaiming_normal_(self.conv_out.weight, mode='fan_out')

        def forward(self, x):
            x = self.conv(x)
            x = self.conv_out(x)
            return x

    class AttentionRefinementModule(nn.Module):
        def __init__(self, in_chan, out_chan):
            super().__init__()
            self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
            self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
            self.bn_atten = nn.BatchNorm2d(out_chan)
            nn.init.kaiming_normal_(self.conv_atten.weight, mode='fan_out')

        def forward(self, x):
            feat = self.conv(x)
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
            atten = self.conv_atten(atten)
            atten = self.bn_atten(atten)
            atten = torch.sigmoid(atten)
            out = feat * atten
            return out

    class FeatureFusionModule(nn.Module):
        def __init__(self, in_chan, out_chan):
            super().__init__()
            self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
            # Channel attention with reduction
            self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out')
            nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out')

        def forward(self, fsp, fcp):
            fcat = torch.cat([fsp, fcp], dim=1)
            feat = self.convblk(fcat)
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
            atten = self.conv1(atten)
            atten = F.relu(atten)
            atten = self.conv2(atten)
            atten = torch.sigmoid(atten)
            feat_atten = feat * atten
            feat_out = feat + feat_atten
            return feat_out

    class BiSeNet(nn.Module):
        def __init__(self, n_classes=19):
            super().__init__()

            # ResNet18 backbone
            import torchvision.models as models
            resnet = models.resnet18(weights=None)
            self.conv1 = resnet.conv1      # 3 → 64
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1    # 64 → 64
            self.layer2 = resnet.layer2    # 64 → 128 (feat8, 1/8 scale)
            self.layer3 = resnet.layer3    # 128 → 256 (feat16, 1/16 scale)
            self.layer4 = resnet.layer4    # 256 → 512 (feat32, 1/32 scale)

            # Context Path components
            self.arm16 = AttentionRefinementModule(256, 128)
            self.arm32 = AttentionRefinementModule(512, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

            # Feature Fusion Module: 128 (spatial) + 128 (context) = 256
            self.ffm = FeatureFusionModule(256, 256)

            # Output heads
            self.conv_out = BiSeNetOutput(256, 256, n_classes)
            self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
            self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

        def forward(self, x):
            H, W = x.shape[2:]

            # Backbone feature extraction
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            feat8 = self.layer2(x)      # 128 channels, 1/8 scale
            feat16 = self.layer3(feat8)  # 256 channels, 1/16 scale
            feat32 = self.layer4(feat16) # 512 channels, 1/32 scale

            # Context Path
            avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
            avg = self.conv_avg(avg)
            avg_up = F.interpolate(avg, size=feat32.shape[2:], mode='nearest')

            feat32_arm = self.arm32(feat32)
            feat32_sum = feat32_arm + avg_up
            feat32_up = F.interpolate(feat32_sum, size=feat16.shape[2:], mode='nearest')
            feat32_up = self.conv_head32(feat32_up)

            feat16_arm = self.arm16(feat16)
            feat16_sum = feat16_arm + feat32_up
            feat16_up = F.interpolate(feat16_sum, size=feat8.shape[2:], mode='nearest')
            feat16_up = self.conv_head16(feat16_up)

            # Spatial Path: use feat8 directly (128 channels)
            # In original face-parsing.PyTorch, they use feat8 from ResNet as spatial features
            feat_sp = feat8  # 128 channels

            # Feature Fusion: concat(128, 128) = 256
            feat_fuse = self.ffm(feat_sp, feat16_up)

            # Main output
            feat_out = self.conv_out(feat_fuse)
            feat_out = F.interpolate(feat_out, size=(H, W), mode='bilinear', align_corners=True)

            # Only return main output for inference
            return feat_out

    return BiSeNet(n_classes=19)


# ==============================================================================
# PyTorch → ONNX Export
# ==============================================================================
def export_to_onnx(pth_path, onnx_path, input_size=(512, 512)):
    """Export BiSeNet PyTorch model to ONNX format."""
    print(f"\n[EXPORT] PyTorch → ONNX")
    print(f"   Input:  {pth_path}")
    print(f"   Output: {onnx_path}")
    print(f"   Size:   {input_size[0]}x{input_size[1]}")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Build model
    print("   [1/4] Building BiSeNet architecture...")
    model = get_bisenet_model()
    model = model.to(device)
    model.eval()

    # Load weights
    print("   [2/4] Loading checkpoint...")
    try:
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        print(f"   [INFO] Checkpoint has {len(state_dict)} keys")

        # Remove 'module.' prefix if present (from DataParallel)
        # And map keys from original face-parsing.PyTorch format to our format
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k

            # Remove module. prefix
            if new_k.startswith('module.'):
                new_k = new_k[7:]

            # Map cp.resnet.* to direct backbone layers
            # Original: cp.resnet.conv1 -> Our: conv1
            if new_k.startswith('cp.resnet.'):
                new_k = new_k.replace('cp.resnet.', '')

            # Map cp.arm* to arm*
            elif new_k.startswith('cp.'):
                new_k = new_k[3:]  # Remove 'cp.'

            new_state_dict[new_k] = v

        # Show some key examples for debugging
        sample_keys = list(new_state_dict.keys())[:5]
        print(f"   [INFO] Sample checkpoint keys (after mapping): {sample_keys}")

        # Get model keys for comparison
        model_keys = list(model.state_dict().keys())[:5]
        print(f"   [INFO] Model expects: {model_keys}")

        # Load with strict=False to handle minor mismatches
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        if missing:
            print(f"   [WARN] Missing keys: {len(missing)}")
            if len(missing) <= 10:
                for k in missing:
                    print(f"      - {k}")
            else:
                print(f"      (showing first 5)")
                for k in missing[:5]:
                    print(f"      - {k}")

        if unexpected:
            print(f"   [WARN] Unexpected keys: {len(unexpected)}")
            if len(unexpected) <= 10:
                for k in unexpected:
                    print(f"      - {k}")
            else:
                print(f"      (showing first 5)")
                for k in unexpected[:5]:
                    print(f"      - {k}")

        # Check if any weights were actually loaded
        loaded_count = len(new_state_dict) - len(unexpected)
        if loaded_count == 0:
            print(f"   [ERROR] No weights were loaded! Key mismatch.")
            return False

        print(f"   [OK] Loaded {loaded_count}/{len(model.state_dict())} weights")

    except Exception as e:
        print(f"   [ERROR] Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Export to ONNX
    print("   [3/4] Exporting to ONNX...")
    try:
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)

        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=13,
            do_constant_folding=True
        )

        print(f"   [OK] ONNX exported: {onnx_path}")

    except Exception as e:
        print(f"   [ERROR] ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify ONNX
    print("   [4/4] Verifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"   [OK] ONNX verification passed")
        return True
    except ImportError:
        print("   [WARN] onnx package not installed, skipping verification")
        return True
    except Exception as e:
        print(f"   [WARN] ONNX verification failed: {e}")
        return True  # Continue anyway


# ==============================================================================
# ONNX → TensorRT Conversion
# ==============================================================================
def convert_to_tensorrt(onnx_path, engine_path, input_shape=(512, 512), use_fp16=True):
    """Convert ONNX model to TensorRT engine."""
    print(f"\n[BUILD] ONNX → TensorRT")
    print(f"   Input:  {onnx_path}")
    print(f"   Output: {engine_path}")
    print(f"   Shape:  1x3x{input_shape[0]}x{input_shape[1]}")
    print(f"   FP16:   {'Enabled' if use_fp16 else 'Disabled'}")

    # Setup CUDA environment
    try:
        from src.utils.cuda_helper import setup_cuda_environment
        setup_cuda_environment()
    except:
        pass

    try:
        import tensorrt as trt
        print(f"   TensorRT Version: {trt.__version__}")
    except ImportError:
        print("[ERROR] TensorRT not found. Please install tensorrt package.")
        return False

    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX file not found: {onnx_path}")
        return False

    # Create TensorRT builder
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print("   [1/4] Parsing ONNX model...")
    success = parser.parse_from_file(onnx_path)
    if not success:
        print("[ERROR] ONNX Parse Failed:")
        for i in range(parser.num_errors):
            print(f"   {parser.get_error(i)}")
        return False

    print(f"   [2/4] Network parsed successfully")
    print(f"         Inputs: {network.num_inputs}")
    print(f"         Outputs: {network.num_outputs}")

    # Configure optimization profile
    h, w = input_shape
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"         Input Name: {input_name}")

    profile.set_shape(input_name, (1, 3, h, w), (1, 3, h, w), (1, 3, h, w))
    config.add_optimization_profile(profile)

    # Enable FP16
    print("   [3/4] Configuring build options...")
    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("         FP16 acceleration enabled")
    else:
        print("         Using FP32 precision")

    # Set workspace
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        print("         Workspace: 1GB")
    except:
        try:
            config.max_workspace_size = 1 << 30
        except:
            pass

    # Build engine
    print("   [4/4] Building TensorRT engine (this may take 2-5 minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("[ERROR] Engine build failed!")
        return False

    os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"\n[SUCCESS] TensorRT engine saved!")
    print(f"   Path: {engine_path}")
    print(f"   Size: {engine_size_mb:.1f} MB")

    return True


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="BiSeNet V2 Full Conversion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (download first with: python tools/download_models.py)
    python tools/convert_bisenet.py

    # Export PyTorch to ONNX only
    python tools/convert_bisenet.py --export-only

    # Build TensorRT from existing ONNX only
    python tools/convert_bisenet.py --build-only

    # Custom paths
    python tools/convert_bisenet.py --pth my_model.pth --onnx my_model.onnx --engine my_model.engine
        """
    )

    parser.add_argument('--pth', type=str, default=PTH_PATH,
                       help=f"PyTorch checkpoint path (default: {PTH_PATH})")
    parser.add_argument('--onnx', type=str, default=ONNX_PATH,
                       help=f"ONNX output path (default: {ONNX_PATH})")
    parser.add_argument('--engine', type=str, default=ENGINE_PATH,
                       help=f"TensorRT engine path (default: {ENGINE_PATH})")
    parser.add_argument('--height', type=int, default=512,
                       help="Input height (default: 512)")
    parser.add_argument('--width', type=int, default=512,
                       help="Input width (default: 512)")
    parser.add_argument('--no-fp16', action='store_true',
                       help="Disable FP16 precision")
    parser.add_argument('--export-only', action='store_true',
                       help="Only export PyTorch to ONNX")
    parser.add_argument('--build-only', action='store_true',
                       help="Only build TensorRT from ONNX")

    args = parser.parse_args()

    print("=" * 60)
    print("   MUSE BiSeNet V2 Converter")
    print("   Face Parsing Model for SkinParser")
    print("=" * 60)

    input_size = (args.height, args.width)

    # Check for existing engine
    if os.path.exists(args.engine) and not args.export_only:
        print(f"\n[INFO] Engine already exists: {args.engine}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("Conversion cancelled.")
            return

    # Step 1: Export to ONNX (unless build-only)
    if not args.build_only:
        if not os.path.exists(args.pth):
            print(f"\n[ERROR] PyTorch checkpoint not found: {args.pth}")
            print("\nPlease download the model first:")
            print("   python tools/download_models.py")
            sys.exit(1)

        if not export_to_onnx(args.pth, args.onnx, input_size):
            print("\n[FAILED] ONNX export failed!")
            sys.exit(1)

    if args.export_only:
        print("\n[DONE] ONNX export complete!")
        return

    # Step 2: Build TensorRT engine
    if not os.path.exists(args.onnx):
        print(f"\n[ERROR] ONNX file not found: {args.onnx}")
        print("Run without --build-only to export ONNX first.")
        sys.exit(1)

    if not convert_to_tensorrt(args.onnx, args.engine, input_size, not args.no_fp16):
        print("\n[FAILED] TensorRT build failed!")
        sys.exit(1)

    # Clean up ONNX (optional)
    # if os.path.exists(args.onnx):
    #     os.remove(args.onnx)

    print("\n" + "=" * 60)
    print("[DONE] BiSeNet V2 conversion complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run MUSE to test SkinParser integration")
    print("2. SkinParser will automatically use the new engine")


if __name__ == "__main__":
    main()
