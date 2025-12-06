# Project MUSE - convert_student_to_trt.py
# Updates: Handles separate Seg/Pose models & Single Profile Support
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import torch
import tensorrt as trt
import glob
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ai.distillation.student.model_arch import MuseStudentModel

TARGET_W = 960
TARGET_H = 544

def convert_all(target_profile=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "assets", "models", "personal")
    
    # Define patterns based on target
    if target_profile:
        print(f"[FILTER] Converting ONLY profile: '{target_profile}'")
        seg_pattern = f"student_seg_{target_profile}.pth"
        pose_pattern = f"student_pose_{target_profile}.pth"
    else:
        seg_pattern = "student_seg_*.pth"
        pose_pattern = "student_pose_*.pth"

    seg_files = glob.glob(os.path.join(model_dir, seg_pattern))
    pose_files = glob.glob(os.path.join(model_dir, pose_pattern))
    
    all_files = seg_files + pose_files
    
    if not all_files:
        print(f"[ERROR] No models(.pth) found for conversion. (Pattern: {seg_pattern}/{pose_pattern})")
        return

    total = len(all_files)
    print(f"[LOOP] Found {total} models to convert.")

    for i, pth in enumerate(all_files):
        filename = os.path.basename(pth)
        print(f"\n[{i+1}/{total}] Processing: {filename}")
        
        # Determine mode
        if "student_seg_" in filename:
            mode = "seg"
        elif "student_pose_" in filename:
            mode = "pose"
        else:
            continue

        onnx_path = pth.replace(".pth", ".onnx")
        engine_path = pth.replace(".pth", ".engine")
        
        base_progress = int((i / total) * 100)
        print(f"[PROGRESS] {base_progress}")
        
        try:
            export_onnx(pth, onnx_path, mode)
            
            print(f"[PROGRESS] {base_progress + int(50/total)}")
            build_engine(onnx_path, engine_path)
        except Exception as e:
            print(f"[ERROR] Failed to convert {filename}: {e}")

    print("[PROGRESS] 100")
    print("\n[OK] Conversion complete.")

def export_onnx(pth_path, onnx_path, mode):
    print(f"   -> Exporting to ONNX ({mode.upper()})...")
    device = torch.device("cuda")
    
    model = MuseStudentModel(num_keypoints=17, pretrained=False, mode=mode).to(device)
    model.load_state_dict(torch.load(pth_path))
    model.eval()

    dummy_input = torch.randn(1, 3, TARGET_H, TARGET_W).to(device)
    
    output_names = ['seg'] if mode == 'seg' else ['pose']
    dynamic_axes = {'input': {0: 'B'}, output_names[0]: {0: 'B'}}

    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17 
    )
    print("   [OK] ONNX Exported")

def build_engine(onnx_path, engine_path):
    print("   -> Building TensorRT Engine...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("   [ERROR] ONNX Parse Failed")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    input_shape = (1, 3, TARGET_H, TARGET_W)
    profile = builder.create_optimization_profile()
    profile.set_shape("input", input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)
    
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
    except:
        pass

    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("   [OK] Engine Built")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None, help="Target profile to convert")
    args = parser.parse_args()
    
    convert_all(args.profile)