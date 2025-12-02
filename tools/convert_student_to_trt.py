# Project MUSE - convert_student_to_trt.py
# Target: Convert Student Model (ResNet-34 U-Net) to TensorRT Engine
# Resolution: 960x544 (High-Fidelity)
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import torch
import torch.onnx
import tensorrt as trt

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.distillation.student.model_arch import MuseStudentModel

# [High-Fidelity Resolution Config]
# Width: 960, Height: 544
TARGET_W = 960
TARGET_H = 544

def export_onnx(pth_path, onnx_path):
    print(f"🚀 [Step 1] PyTorch -> ONNX 변환 시작... (Res: {TARGET_W}x{TARGET_H})")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MuseStudentModel(num_keypoints=17).to(device)
    
    # 가중치 로드
    if not os.path.exists(pth_path):
        print(f"❌ 모델 파일이 없습니다: {pth_path}")
        print("   -> 먼저 'tools/train_student.py'를 실행하여 모델을 학습하세요.")
        sys.exit(1)
        
    try:
        state_dict = torch.load(pth_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("   ✅ PyTorch 모델 로드 완료")
    except Exception as e:
        print(f"❌ 가중치 로드 실패: {e}")
        sys.exit(1)

    # 더미 입력 (Batch:1, Channel:3, Height:544, Width:960)
    # 주의: PyTorch는 (N, C, H, W) 순서입니다.
    dummy_input = torch.randn(1, 3, TARGET_H, TARGET_W).to(device)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['seg_logits', 'pose_heatmaps'],
            # 배치 사이즈는 가변(dynamic)으로 두거나 고정할 수 있습니다. 여기선 1로 고정 추천(RT 성능 최적화)
            dynamic_axes={'input': {0: 'batch_size'}, 'seg_logits': {0: 'batch_size'}, 'pose_heatmaps': {0: 'batch_size'}},
            opset_version=13
        )
        print(f"   ✅ ONNX 추출 완료: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        sys.exit(1)

def build_engine(onnx_path, engine_path):
    print(f"🚀 [Step 2] ONNX -> TensorRT Engine 빌드 시작...")
    
    # CUDA 환경 설정 (필요 시)
    try:
        from src.utils.cuda_helper import setup_cuda_environment
        setup_cuda_environment()
    except:
        pass

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    
    # Explicit Batch 플래그 설정
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # ONNX 파싱
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX 파일이 없습니다: {onnx_path}")
        sys.exit(1)

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ ONNX 파싱 실패")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(1)

    # 최적화 프로파일 설정 (입력 크기 고정: 960x544)
    # TensorRT Shape: (Batch, Channel, Height, Width)
    input_shape = (1, 3, TARGET_H, TARGET_W)
    
    profile = builder.create_optimization_profile()
    profile.set_shape("input", input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)

    # 메모리 풀 설정 (4GB)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    except AttributeError:
        config.max_workspace_size = 1 << 32

    # FP16 가속 활성화 (RTX 3060 지원)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("   ✨ FP16 고속 연산 모드 활성화")

    # 엔진 빌드
    print("   ⏳ 엔진 빌드 중... (약 1~2분 소요)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("❌ 엔진 빌드 실패")
        sys.exit(1)

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"   ✅ 엔진 저장 완료: {engine_path}")

def main():
    print("========================================================")
    print("   MUSE Student Model Optimization Tool")
    print("   (High-Fidelity Mode: 960x544)")
    print("========================================================")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "assets", "models", "personal")
    
    # 여러 프로파일을 순회하며 변환하도록 확장 가능하지만, 일단 기본 파일명 기준
    # 실제 운영 시에는 인자값으로 파일명을 받아야 함
    
    pth_path = os.path.join(model_dir, "student_model_final.pth")
    onnx_path = os.path.join(model_dir, "student_model.onnx")
    engine_path = os.path.join(model_dir, "student_model.engine")
    
    # 1. Export ONNX
    export_onnx(pth_path, onnx_path)
    
    # 2. Build TensorRT Engine
    build_engine(onnx_path, engine_path)
    
    print("\n🎉 변환 완료! 이제 'tools/run_muse.py'를 실행하면 고화질 추론이 작동합니다.")

if __name__ == "__main__":
    main()