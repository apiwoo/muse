# Project MUSE - convert_student_to_trt.py
# Transformer (SegFormer) Support
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import torch
import tensorrt as trt
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ai.distillation.student.model_arch import MuseStudentModel

TARGET_W = 960
TARGET_H = 544

def convert_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "assets", "models", "personal")
    
    # Scan all .pth files
    pth_files = glob.glob(os.path.join(model_dir, "student_*.pth"))
    
    if not pth_files:
        print("❌ 변환할 모델(.pth)이 없습니다.")
        return

    for pth in pth_files:
        print(f"\n🔄 Converting: {os.path.basename(pth)}")
        onnx_path = pth.replace(".pth", ".onnx")
        engine_path = pth.replace(".pth", ".engine")
        
        export_onnx(pth, onnx_path)
        build_engine(onnx_path, engine_path)

def export_onnx(pth_path, onnx_path):
    device = torch.device("cuda")
    model = MuseStudentModel(num_keypoints=17, pretrained=False).to(device)
    model.load_state_dict(torch.load(pth_path))
    model.eval()

    dummy_input = torch.randn(1, 3, TARGET_H, TARGET_W).to(device)
    
    # SegFormer has complex operations (Reshape, Permute), needs higher opset
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'],
        output_names=['seg', 'pose'],
        dynamic_axes={'input': {0: 'B'}, 'seg': {0: 'B'}, 'pose': {0: 'B'}},
        opset_version=17 # Important for Transformer
    )
    print("   ✅ ONNX Exported")

def build_engine(onnx_path, engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("   ❌ ONNX Parse Failed")
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
    print("   ✅ TensorRT Engine Built")

if __name__ == "__main__":
    convert_all()