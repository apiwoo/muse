# Project MUSE - merge_lora_to_engine.py
# Merges LoRA weights into Base Model and converts to TensorRT
# Updated: Support for both Pose (ViTPose) and Seg (MODNet)
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import torch
import argparse
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ai.models.vitpose_lora import ViTPoseLoRA
from src.ai.models.modnet_lora import MODNetLoRA

def merge_and_convert(profile_name, mode='pose'):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "assets", "models")
    personal_dir = os.path.join(model_dir, "personal")
    
    if mode == 'pose':
        return _merge_pose(profile_name, base_dir, model_dir, personal_dir)
    elif mode == 'seg':
        return _merge_seg(profile_name, base_dir, model_dir, personal_dir)
    else:
        print(f"[ERROR] Unknown mode: {mode}")
        return False

def _merge_pose(profile_name, base_dir, model_dir, personal_dir):
    base_pth = os.path.join(model_dir, "tracking", "vitpose_base_coco_256x192.pth")
    lora_pth = os.path.join(personal_dir, f"vitpose_lora_weights_{profile_name}.pth")
    
    if not os.path.exists(lora_pth):
        print(f"[ERROR] LoRA weights not found: {lora_pth}")
        return False

    print(f"[MERGE] Merging Pose LoRA for profile: {profile_name}...")
    
    # 1. Load Base
    model = ViTPoseLoRA(img_size=(256, 192), patch_size=16, embed_dim=768, depth=12)
    checkpoint = torch.load(base_pth, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('backbone.', '')
        if 'keypoint_head.deconv_layers.0.' in new_k: new_k = new_k.replace('keypoint_head.deconv_layers.0.', 'keypoint_head.0.')
        elif 'keypoint_head.deconv_layers.1.' in new_k: new_k = new_k.replace('keypoint_head.deconv_layers.1.', 'keypoint_head.1.')
        elif 'keypoint_head.deconv_layers.3.' in new_k: new_k = new_k.replace('keypoint_head.deconv_layers.3.', 'keypoint_head.3.')
        elif 'keypoint_head.deconv_layers.4.' in new_k: new_k = new_k.replace('keypoint_head.deconv_layers.4.', 'keypoint_head.4.')
        elif 'keypoint_head.final_layer.' in new_k: new_k = new_k.replace('keypoint_head.final_layer.', 'keypoint_head.6.')
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict, strict=False)
    
    # 2. Inject & Load LoRA
    model.inject_lora(rank=8)
    lora_dict = torch.load(lora_pth, map_location='cpu')
    model.load_state_dict(lora_dict, strict=False)
    
    # 3. Fold
    _fold_lora_weights(model)

    # 4. Export ONNX
    onnx_path = os.path.join(personal_dir, f"vitpose_lora_{profile_name}.onnx")
    dummy_input = torch.randn(1, 3, 256, 192)
    _export_onnx(model, dummy_input, onnx_path)
    
    # 5. Build Engine
    engine_path = os.path.join(personal_dir, f"vitpose_lora_{profile_name}.engine")
    return _build_engine(base_dir, onnx_path, engine_path)

def _merge_seg(profile_name, base_dir, model_dir, personal_dir):
    base_pth = os.path.join(model_dir, "segmentation", "modnet_webcam_portrait_matting.ckpt")
    lora_pth = os.path.join(personal_dir, f"modnet_lora_weights_{profile_name}.pth")
    
    if not os.path.exists(lora_pth):
        print(f"[ERROR] Seg LoRA weights not found: {lora_pth}")
        return False

    print(f"[MERGE] Merging Seg LoRA for profile: {profile_name}...")
    
    # 1. Load Base
    model = MODNetLoRA(in_channels=3)
    if os.path.exists(base_pth):
        ckpt = torch.load(base_pth, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        new_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_dict, strict=False)
    
    # 2. Inject & Load LoRA
    model.inject_lora(rank=4)
    lora_dict = torch.load(lora_pth, map_location='cpu')
    model.load_state_dict(lora_dict, strict=False)
    
    # 3. Fold LoRA
    print("   -> Folding LoRA weights into Conv2d layers...")
    model.eval()
    
    # Access injected layers
    for lora_layer in model.lora_layers:
        # Update = (B @ A) * scaling
        # lora_A: (rank, in, k, k), lora_B: (out, rank, 1, 1)
        # Note: In MODNetLoRA, we simplified LoRA_B to 1x1.
        # Convolution of weights: W_new = W + (W_B * W_A)
        # Since W_B is 1x1, this is effectively matrix mult on channel dim.
        
        w_a = lora_layer.lora_A.weight # (r, in, k, k)
        w_b = lora_layer.lora_B.weight # (out, r, 1, 1)
        
        # Calculate B * A efficiently
        # Reshape A to (r, in*k*k)
        r, inc, k, _ = w_a.shape
        outc = w_b.shape[0]
        
        flat_a = w_a.view(r, -1) 
        flat_b = w_b.view(outc, r)
        
        # (out, r) @ (r, in*k*k) -> (out, in*k*k)
        update = (flat_b @ flat_a).view(outc, inc, k, k)
        update *= lora_layer.scaling
        
        # Add to original conv weight
        with torch.no_grad():
            lora_layer.conv.weight.add_(update)
            
        # Restore original conv (remove wrapper)
        # Note: This part requires replacing the module in the parent. 
        # But since we only need to export state_dict or ONNX from 'model', 
        # and 'model.forward' uses 'self.conv', we can just disable LoRA part in forward 
        # or rely on the fact that ONNX tracer will trace the math.
        # Ideally, we should strip the wrapper. For now, we rely on ONNX tracing.
        # To ensure clean export, we set LoRA weights to zero after folding? 
        # No, that would break if we run it.
        # Best way: LoRAConv2d.forward adds LoRA term. If we folded it into self.conv, we should disable LoRA term.
        lora_layer.scaling = 0.0 # Disable LoRA term addition

    # 4. Export ONNX (Target Resolution 544p)
    onnx_path = os.path.join(personal_dir, f"modnet_lora_{profile_name}.onnx")
    # MODNet needs fixed multiple of 32
    dummy_input = torch.randn(1, 3, 544, 960) 
    _export_onnx(model, dummy_input, onnx_path)
    
    # 5. Build Engine
    engine_path = os.path.join(personal_dir, f"modnet_lora_{profile_name}.engine")
    return _build_engine(base_dir, onnx_path, engine_path)

def _fold_lora_weights(model):
    print("   -> Folding LoRA weights into Base Linear layers...")
    model.eval()
    for i, blk in enumerate(model.blocks):
        lora_layer = blk.attn.qkv
        update = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
        with torch.no_grad():
            lora_layer.linear.weight.add_(update)
        blk.attn.qkv = lora_layer.linear

def _export_onnx(model, input_tensor, path):
    torch.onnx.export(
        model, input_tensor, path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'B'}, 'output': {0: 'B'}},
        opset_version=13
    )
    print(f"   -> ONNX Exported: {path}")

def _build_engine(base_dir, onnx_path, engine_path):
    print(f"   -> Building TensorRT Engine...")
    cmd = [
        sys.executable, 
        os.path.join(base_dir, "tools", "trt_converter.py"),
        "--build-worker", onnx_path, engine_path
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] High-Precision Engine Created: {os.path.basename(engine_path)}")
        if os.path.exists(onnx_path): os.remove(onnx_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Engine build failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mode", default='pose', choices=['pose', 'seg'])
    args = parser.parse_args()
    merge_and_convert(args.profile, args.mode)