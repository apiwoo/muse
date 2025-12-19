# Project MUSE - trt_converter.py
# Target: RTX 3060/4090 Mode A (High Performance)
# Updated: Support dynamic input shape via arguments
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import subprocess
import time
import argparse

# ==================================================================================
# [Mode 1] PyTorch Worker (ONNX Export)
# ==================================================================================
def run_export_worker(pth_path, onnx_path, variant):
    print(f"[START] [Process 1] PyTorch Worker Started (Variant: {variant.upper()})...")
    
    import torch
    import torch.nn as nn
    
    # --------------------------------------------------------
    # ViTPose Configs (Huge Only)
    # --------------------------------------------------------
    MODEL_CONFIGS = {
        'huge': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16},
    }
    
    cfg = MODEL_CONFIGS.get(variant.lower())
    if not cfg:
        print(f"[ERROR] Unknown variant: {variant}")
        sys.exit(1)

    class PatchEmbed(nn.Module):
        def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=1280):
            super().__init__()
            self.img_size = img_size
            self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
            self.num_patches = self.grid_size[0] * self.grid_size[1]
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        def forward(self, x):
            x = self.proj(x)
            x = x.flatten(2).transpose(1, 2)
            return x

    class Attention(nn.Module):
        def __init__(self, dim, num_heads=16, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
            super().__init__()
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)
        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    class Block(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class ViTPose(nn.Module):
        def __init__(self, img_size=(256, 192), patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4., num_classes=17):
            super().__init__()
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)) 
            self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)])
            self.norm = nn.LayerNorm(embed_dim)
            self.keypoint_head = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
            )

        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x) 
            pos_embed = self.pos_embed[:, 1:, :] 
            x = x + pos_embed
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            H, W = self.patch_embed.grid_size
            x = x.transpose(1, 2).reshape(B, -1, H, W)
            x = self.keypoint_head(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using Device: {device}, Config: {cfg}")

    model = ViTPose(**cfg).to(device)
    model.eval()

    try:
        print("   [LOAD] Loading checkpoint...")
        checkpoint = torch.load(pth_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        new_state_dict = {}
        mapped_count = 0
        
        for k, v in state_dict.items():
            new_k = k.replace('backbone.', '')
            if 'cls_token' in new_k: continue
            
            # Head Weight Remapping
            if 'keypoint_head.deconv_layers.0.' in new_k:
                new_k = new_k.replace('keypoint_head.deconv_layers.0.', 'keypoint_head.0.')
            elif 'keypoint_head.deconv_layers.1.' in new_k:
                new_k = new_k.replace('keypoint_head.deconv_layers.1.', 'keypoint_head.1.')
            elif 'keypoint_head.deconv_layers.3.' in new_k:
                new_k = new_k.replace('keypoint_head.deconv_layers.3.', 'keypoint_head.3.')
            elif 'keypoint_head.deconv_layers.4.' in new_k:
                new_k = new_k.replace('keypoint_head.deconv_layers.4.', 'keypoint_head.4.')
            elif 'keypoint_head.final_layer.' in new_k:
                new_k = new_k.replace('keypoint_head.final_layer.', 'keypoint_head.6.')

            new_state_dict[new_k] = v
            mapped_count += 1

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        print(f"   [OK] Weights Loaded: {mapped_count} keys mapped.")
        if len(missing_keys) > 0:
            head_missing = [k for k in missing_keys if 'keypoint_head' in k]
            if head_missing:
                print(f"   [ERROR] [CRITICAL] Head weights missing: {head_missing}")
                sys.exit(1)
        
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        sys.exit(1)

    dummy_input = torch.randn(1, 3, 256, 192).to(device)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=13  
        )
        print(f"   [OK] ONNX Saved: {onnx_path}")
    except Exception as e:
        print(f"[ERROR] ONNX Export Failed: {e}")
        sys.exit(1)

# ==================================================================================
# [Mode 2] TensorRT Worker (Engine Build)
# ==================================================================================
def run_build_worker(onnx_path, engine_path, input_shape=(256, 192)):
    """
    input_shape: (height, width)
    """
    print(f"[START] [Process 2] TensorRT Worker Started...")
    print(f"   Target Input Shape: 1x3x{input_shape[0]}x{input_shape[1]}")
    
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils.cuda_helper import setup_cuda_environment
        setup_cuda_environment()
    except: pass

    import tensorrt as trt
    print(f"   TensorRT Version: {trt.__version__}")

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX file not found: {onnx_path}")
        sys.exit(1)

    success = parser.parse_from_file(onnx_path)
    if not success:
        print("[ERROR] ONNX Parse Failed")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        sys.exit(1)

    # Dynamic Shape Configuration
    h, w = input_shape
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, h, w), (1, 3, h, w), (1, 3, h, w))
    config.add_optimization_profile(profile)
    
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    except: pass

    print("   [WAIT] Building Engine (This may take 3-5 mins)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("[ERROR] Engine Build Failed")
        sys.exit(1)
        
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"   [OK] Engine Saved: {engine_path}")

# ==================================================================================
# [Main Manager]
# ==================================================================================
def main():
    # Worker Logic Checks via sys.argv manually to avoid conflicts with argparse in subprocess calls
    if len(sys.argv) > 1:
        if sys.argv[1] == '--export-worker':
            # Usage: script.py --export-worker pth onnx variant
            if len(sys.argv) < 5:
                print("Usage: --export-worker <pth> <onnx> <variant>")
                sys.exit(1)
            run_export_worker(sys.argv[2], sys.argv[3], sys.argv[4])
            return
            
        elif sys.argv[1] == '--build-worker':
            # Usage: script.py --build-worker onnx engine [--input-shape H W]
            if len(sys.argv) < 4:
                print("Usage: --build-worker <onnx> <engine> [--input-shape H W]")
                sys.exit(1)
                
            onnx_path = sys.argv[2]
            engine_path = sys.argv[3]
            
            # Default to Pose Shape
            height = 256
            width = 192
            
            # Parse optional shape
            if '--input-shape' in sys.argv:
                try:
                    idx = sys.argv.index('--input-shape')
                    if idx + 2 < len(sys.argv):
                        height = int(sys.argv[idx+1])
                        width = int(sys.argv[idx+2])
                except ValueError:
                    print("[WARN] Invalid input shape format. Using default.")

            run_build_worker(onnx_path, engine_path, (height, width))
            return

    # If run directly (Manager Mode for default conversion)
    print("========================================================")
    print("   MUSE ViTPose Converter (Huge Model Only)")
    print("========================================================")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "assets", "models", "tracking")
    
    # Define Targets (Huge Only)
    targets = [
        {
            'name': 'Huge (Runtime)',
            'variant': 'huge',
            'pth': os.path.join(MODEL_DIR, "vitpose_huge_coco_256x192.pth"),
            'onnx': os.path.join(MODEL_DIR, "vitpose_huge.onnx"),
            'engine': os.path.join(MODEL_DIR, "vitpose_huge.engine")
        }
    ]

    for t in targets:
        print(f"\n[{t['name']}] Checking...")
        if not os.path.exists(t['pth']):
            print(f"   [SKIP] Model source not found: {os.path.basename(t['pth'])}")
            continue
            
        if os.path.exists(t['engine']):
            print(f"   [SKIP] Engine already exists: {os.path.basename(t['engine'])}")
            continue

        print(f"   [START] Converting {t['variant'].upper()}...")
        
        # 1. Export ONNX
        try:
            subprocess.run([sys.executable, __file__, '--export-worker', t['pth'], t['onnx'], t['variant']], check=True)
        except subprocess.CalledProcessError:
            print("   [ERROR] Export Failed. Skipping.")
            continue
            
        # 2. Build Engine (Default shape for ViTPose is 256x192)
        try:
            subprocess.run([sys.executable, __file__, '--build-worker', t['onnx'], t['engine'], '--input-shape', '256', '192'], check=True)
            print("   [SUCCESS] Conversion Success!")
        except subprocess.CalledProcessError:
            print("   [ERROR] Build Failed. Skipping.")
            
        # Clean ONNX
        if os.path.exists(t['onnx']): os.remove(t['onnx'])

    print("\n[DONE] All tasks finished.")

if __name__ == "__main__":
    main()