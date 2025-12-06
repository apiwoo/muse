# Project MUSE - convert_models_to_trt.py
# High-Fidelity TensorRT Converter (Dynamic Shape for 1080p)
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import tensorrt as trt
import argparse

# Check TensorRT Version
print(f"🚀 TensorRT Version: {trt.__version__}")

class TRTBuilder:
    def __init__(self, verbose=False):
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, "")
        
    def build_engine(self, onnx_path, engine_path, input_name='input', 
                     min_shape=(1, 3, 512, 512), 
                     opt_shape=(1, 3, 1080, 1920), 
                     max_shape=(1, 3, 1080, 1920),
                     fp16=True):
        """
        Builds a TensorRT engine with Dynamic Shapes to support native 1080p processing.
        """
        print(f"\n================================================================")
        print(f"   Building Engine: {os.path.basename(engine_path)}")
        print(f"   Input ONNX: {onnx_path}")
        print(f"   Resolution Target: {opt_shape[3]}x{opt_shape[2]} (Opt)")
        print(f"================================================================")

        if not os.path.exists(onnx_path):
            print(f"❌ [ERROR] ONNX file not found: {onnx_path}")
            return False

        builder = trt.Builder(self.logger)
        # Explicit Batch Flag is required for Dynamic Shapes
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, self.logger)

        # 1. Parse ONNX
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("❌ [ERROR] ONNX Parse Failed:")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return False
        
        print("   ✅ ONNX Parsed Successfully.")

        # 2. Optimization Profile (Dynamic Shape)
        # This is CRITICAL for 1080p Accuracy.
        # We tell TRT: "Optimize for 1080p, but allow down to 512p if needed."
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        print(f"   🔧 Optimization Profile Set: Min={min_shape}, Opt={opt_shape}, Max={max_shape}")

        # 3. FP16 Mode (Speed up without losing visible accuracy)
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("   ✨ FP16 Acceleration Enabled.")
        else:
            print("   ⚠️ FP16 Not supported or disabled. Using FP32.")

        # 4. Memory Pool (Allow huge workspace for high-res layers)
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2GB
        except:
            config.max_workspace_size = 2 << 30

        # 5. Build Engine
        print("   ⏳ Building Engine... (This may take a few minutes for 1080p optimization)")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print("❌ [ERROR] Engine Build Failed.")
            return False

        # 6. Save
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        
        print(f"   🎉 Engine Saved: {engine_path}")
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=['all', 'modnet', 'deeplab'], default='all')
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root_dir, "assets", "models")
    
    # Define Models
    models = []
    
    # 1. MODNet (Matting)
    if args.target in ['all', 'modnet']:
        models.append({
            'name': 'MODNet',
            'onnx': os.path.join(model_dir, "segmentation", "modnet.onnx"),
            'engine': os.path.join(model_dir, "segmentation", "modnet_1080p.engine"),
            'input_name': 'input', # Usually 'input' or 'input_1'
            # MODNet ONNX often has fixed input size in export. 
            # If the downloaded ONNX is static 512, this dynamic shape might fail unless the ONNX is dynamic.
            # *Assuming we use a dynamic-ready ONNX or we will fix it.*
            # For this script, we assume the user provided/downloaded a valid ONNX.
            'opt_shape': (1, 3, 1080, 1920) 
        })

    # 2. DeepLabV3+ (Semantic) -> We need to export this from Torch first usually, 
    # but let's assume we have an ONNX or will add an export step later.
    # For now, placeholder or if user has it.
    
    builder = TRTBuilder()

    for m in models:
        # Special check for MODNet input name (it varies)
        # We can inspect ONNX using onnx library if installed, but let's try 'input' default
        
        success = builder.build_engine(
            m['onnx'], m['engine'], 
            input_name=m['input_name'],
            opt_shape=m['opt_shape'],
            max_shape=m['opt_shape'] # Max matches Opt for 1080p focus
        )
        
        if not success:
            print(f"\n⚠️ {m['name']} conversion failed. Please check if ONNX file exists and is valid.")
            print("   (Note: Some static-shape ONNX files cannot be converted to dynamic TRT engines directly.)")

if __name__ == "__main__":
    main()