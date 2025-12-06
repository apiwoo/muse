# Project MUSE - convert_models_to_trt.py
# High-Fidelity TensorRT Converter (Fixed for Stride 32 Compatibility)
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
                     opt_shape=(1, 3, 1088, 1920), 
                     max_shape=(1, 3, 1088, 1920),
                     fp16=True):
        """
        Builds a TensorRT engine with Dynamic Shapes.
        [CRITICAL FIX] Height set to 1088 (multiple of 32) to prevent Concat errors in MODNet.
        """
        print(f"\n================================================================")
        print(f"   Building Engine: {os.path.basename(engine_path)}")
        print(f"   Input ONNX: {onnx_path}")
        print(f"   Target Shape: {opt_shape} (Height aligned to 32)")
        print(f"================================================================")

        if not os.path.exists(onnx_path):
            print(f"❌ [ERROR] ONNX file not found: {onnx_path}")
            return False

        builder = trt.Builder(self.logger)
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

        # 2. Optimization Profile
        # Height 1080 -> 1088 Padding is required at runtime logic, but engine must be 1088.
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        print(f"   🔧 Optimization Profile Set: Opt={opt_shape}")

        # 3. FP16 Mode
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("   ✨ FP16 Acceleration Enabled.")
        else:
            print("   ⚠️ FP16 Not supported or disabled. Using FP32.")

        # 4. Memory Pool
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30) # 4GB for High-Res
        except:
            pass

        # 5. Build Engine
        print("   ⏳ Building Engine... (This make take a while)")
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
    parser.add_argument("--target", choices=['all', 'modnet'], default='all')
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root_dir, "assets", "models")
    
    # Define Models
    models = []
    
    # MODNet (Matting) - Optimized Resolution 960x544 (qHD+)
    if args.target in ['all', 'modnet']:
        models.append({
            'name': 'MODNet',
            'onnx': os.path.join(model_dir, "segmentation", "modnet.onnx"),
            'engine': os.path.join(model_dir, "segmentation", "modnet_544p.engine"), # Output filename updated
            'input_name': 'input',
            # 544 is the nearest multiple of 32 for 540p (960x540)
            'opt_shape': (1, 3, 544, 960) 
        })

    builder = TRTBuilder()

    for m in models:
        success = builder.build_engine(
            m['onnx'], m['engine'], 
            input_name=m['input_name'],
            opt_shape=m['opt_shape'],
            max_shape=m['opt_shape']
        )
        
        if not success:
            print(f"\n⚠️ {m['name']} conversion failed.")

if __name__ == "__main__":
    main()