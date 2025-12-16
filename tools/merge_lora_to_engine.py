# Project MUSE - merge_lora_to_engine.py
# Merges LoRA weights into Base Model and converts to TensorRT
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import torch
import argparse
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ai.models.vitpose_lora import ViTPoseLoRA

def merge_and_convert(profile_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "assets", "models")
    personal_dir = os.path.join(model_dir, "personal")
    
    base_pth = os.path.join(model_dir, "tracking", "vitpose_base_coco_256x192.pth")
    lora_pth = os.path.join(personal_dir, f"vitpose_lora_weights_{profile_name}.pth")
    
    if not os.path.exists(lora_pth):
        print(f"[ERROR] LoRA weights not found: {lora_pth}")
        return False

    print(f"[MERGE] Merging LoRA for profile: {profile_name}...")
    
    # 1. Load Model with Base Weights
    model = ViTPoseLoRA(img_size=(256, 192), patch_size=16, embed_dim=768, depth=12)
    checkpoint = torch.load(base_pth, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Remap Base
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
    
    # 2. Inject LoRA structure
    model.inject_lora(rank=8)
    
    # 3. Load Trained LoRA Weights
    lora_dict = torch.load(lora_pth, map_location='cpu')
    model.load_state_dict(lora_dict, strict=False)
    
    # 4. Merge Weights (Fold LoRA into Linear)
    # W_new = W_base + (B @ A) * scaling
    print("   -> Folding LoRA weights into Base Linear layers...")
    model.eval()
    
    for i, blk in enumerate(model.blocks):
        lora_layer = blk.attn.qkv
        
        # Calculate update matrix
        # A: (rank, in), B: (out, rank) -> B @ A : (out, in)
        update = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
        
        # Add to original weight
        # lora_layer.linear is the original nn.Linear
        with torch.no_grad():
            lora_layer.linear.weight.add_(update)
            
        # Restore original Linear layer (Remove LoRA wrapper for export)
        blk.attn.qkv = lora_layer.linear

    # 5. Export to ONNX
    onnx_path = os.path.join(personal_dir, f"vitpose_lora_{profile_name}.onnx")
    dummy_input = torch.randn(1, 3, 256, 192)
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'B'}, 'output': {0: 'B'}},
        opset_version=13
    )
    print(f"   -> ONNX Exported: {onnx_path}")
    
    # 6. Build TensorRT Engine
    engine_path = os.path.join(personal_dir, f"vitpose_lora_{profile_name}.engine")
    print(f"   -> Building TensorRT Engine...")
    
    # Reusing trt_converter logic via subprocess or import
    # Using subprocess to ensure clean context
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
    args = parser.parse_args()
    merge_and_convert(args.profile)