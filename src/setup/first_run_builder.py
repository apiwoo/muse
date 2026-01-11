# Project MUSE - first_run_builder.py
# First-run TensorRT engine builder module
# Builds GPU-specific TensorRT engines from ONNX models on first run
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys


class FirstRunBuilder:
    """
    Handles first-run TensorRT engine building.

    On first run, checks for required TensorRT engine files and builds
    any missing engines from ONNX models. This ensures the application
    works on any NVIDIA GPU without pre-built engines.
    """

    def __init__(self, project_root):
        """
        Initialize the builder with project root path.

        Args:
            project_root: Path to the project root directory
        """
        self.project_root = project_root
        self.MODEL_DIR = os.path.join(project_root, "assets", "models")

        # Define required engines with their ONNX sources and build parameters
        self.REQUIRED_ENGINES = {
            "modnet": {
                "name": "MODNet Segmentation",
                "onnx_path": os.path.join(self.MODEL_DIR, "segmentation", "modnet.onnx"),
                "engine_path": os.path.join(self.MODEL_DIR, "segmentation", "modnet_544p.engine"),
                "shape": (544, 960)  # (height, width)
            },
            "vitpose": {
                "name": "ViTPose Pose Estimation",
                "onnx_path": os.path.join(self.MODEL_DIR, "tracking", "vitpose_huge.onnx"),
                "engine_path": os.path.join(self.MODEL_DIR, "tracking", "vitpose_huge.engine"),
                "shape": (256, 192)  # (height, width)
            },
            "bisenet": {
                "name": "BiSeNet Skin Parsing",
                "onnx_path": os.path.join(self.MODEL_DIR, "parsing", "bisenet_face_parsing.onnx"),
                "engine_path": os.path.join(self.MODEL_DIR, "parsing", "bisenet_v2_fp16.engine"),
                "shape": (512, 512)  # (height, width)
            }
        }

    def check_engines_exist(self):
        """
        Check if all required TensorRT engine files exist.

        Returns:
            bool: True if all engines exist, False otherwise
        """
        for key, config in self.REQUIRED_ENGINES.items():
            if not os.path.exists(config["engine_path"]):
                return False
        return True

    def get_missing_engines(self):
        """
        Get list of missing engines that need to be built.

        Returns:
            list: List of tuples (name, onnx_path, engine_path, shape) for missing engines
        """
        missing = []
        for key, config in self.REQUIRED_ENGINES.items():
            if not os.path.exists(config["engine_path"]):
                # Check if ONNX source exists
                if os.path.exists(config["onnx_path"]):
                    missing.append((
                        config["name"],
                        config["onnx_path"],
                        config["engine_path"],
                        config["shape"]
                    ))
                else:
                    print(f"[WARNING] ONNX file missing: {config['onnx_path']}")
        return missing

    def check_gpu_available(self):
        """
        Check if NVIDIA GPU and TensorRT are available.

        Returns:
            tuple: (success: bool, error_message: str)
        """
        try:
            # Setup CUDA environment first
            try:
                sys.path.insert(0, os.path.join(self.project_root, "src"))
                from utils.cuda_helper import setup_cuda_environment
                setup_cuda_environment()
            except Exception:
                pass

            import tensorrt as trt

            # Try to create a builder to verify GPU access
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)

            if builder is None:
                return False, "TensorRT Builder creation failed"

            # Check for FP16 support (indicates GPU presence)
            if not builder.platform_has_fast_fp16:
                print("[INFO] GPU does not have fast FP16 support, will use FP32")

            return True, ""

        except ImportError:
            return False, "TensorRT is not installed. Please install tensorrt package."
        except Exception as e:
            return False, f"GPU check failed: {str(e)}"

    def build_single_engine(self, onnx_path, engine_path, shape, progress_callback=None):
        """
        Build a single TensorRT engine from ONNX file.

        Args:
            onnx_path: Path to source ONNX file
            engine_path: Path to output engine file
            shape: Tuple of (height, width) for input shape
            progress_callback: Optional callback function(percent, message)

        Returns:
            bool: True if build successful, False otherwise
        """
        def report_progress(percent, message):
            if progress_callback:
                progress_callback(percent, message)
            print(f"   [{percent:3d}%] {message}")

        report_progress(5, "Initializing TensorRT...")

        try:
            # Setup CUDA environment
            try:
                sys.path.insert(0, os.path.join(self.project_root, "src"))
                from utils.cuda_helper import setup_cuda_environment
                setup_cuda_environment()
            except Exception:
                pass

            import tensorrt as trt

            report_progress(10, "Loading ONNX file...")

            # Check ONNX file exists
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

            # Create TensorRT components
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()
            parser = trt.OnnxParser(network, TRT_LOGGER)

            report_progress(30, "Parsing ONNX model...")

            # Parse ONNX
            success = parser.parse_from_file(onnx_path)
            if not success:
                errors = []
                for i in range(parser.num_errors):
                    errors.append(str(parser.get_error(i)))
                raise RuntimeError(f"ONNX parse failed: {'; '.join(errors)}")

            report_progress(40, "Configuring optimization profile...")

            # Configure optimization profile
            h, w = shape
            profile = builder.create_optimization_profile()

            # Get input tensor name
            input_tensor = network.get_input(0)
            input_name = input_tensor.name

            profile.set_shape(input_name, (1, 3, h, w), (1, 3, h, w), (1, 3, h, w))
            config.add_optimization_profile(profile)

            # Enable FP16 if available
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            # Set workspace memory
            try:
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
            except Exception:
                try:
                    config.max_workspace_size = 1 << 32
                except Exception:
                    pass

            report_progress(50, "Building engine... (2-5 minutes)")

            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                raise RuntimeError("Engine build returned None")

            report_progress(90, "Saving engine file...")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)

            # Save engine
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            report_progress(100, "Complete!")

            return True

        except Exception as e:
            print(f"[ERROR] Engine build failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def build_all_missing(self, progress_callback=None):
        """
        Build all missing TensorRT engines.

        Args:
            progress_callback: Optional callback function(percent, message)

        Returns:
            bool: True if all builds successful, False otherwise
        """
        missing = self.get_missing_engines()

        if not missing:
            if progress_callback:
                progress_callback(100, "All engines already exist!")
            return True

        total = len(missing)
        all_success = True

        for i, (name, onnx_path, engine_path, shape) in enumerate(missing):
            base_progress = (i / total) * 100
            progress_per_engine = 100 / total

            def sub_callback(percent, message):
                overall = base_progress + (percent * progress_per_engine / 100)
                if progress_callback:
                    progress_callback(int(overall), f"[{i+1}/{total}] {name}: {message}")

            print(f"\n[BUILD] Building {name} ({i+1}/{total})")
            print(f"   ONNX: {onnx_path}")
            print(f"   Engine: {engine_path}")
            print(f"   Shape: {shape[0]}x{shape[1]}")

            success = self.build_single_engine(onnx_path, engine_path, shape, sub_callback)

            if not success:
                print(f"[ERROR] Failed to build {name}")
                all_success = False
                # Continue with other engines instead of stopping

        return all_success


def main():
    """Test the FirstRunBuilder."""
    print("=" * 60)
    print("   FirstRunBuilder Test")
    print("=" * 60)

    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    builder = FirstRunBuilder(project_root)

    print("\n[1] Checking existing engines...")
    if builder.check_engines_exist():
        print("   All engines exist!")
    else:
        print("   Some engines are missing.")

    print("\n[2] Missing engines:")
    missing = builder.get_missing_engines()
    if missing:
        for name, onnx, engine, shape in missing:
            print(f"   - {name}")
            print(f"     ONNX: {onnx}")
            print(f"     Engine: {engine}")
    else:
        print("   None!")

    print("\n[3] Checking GPU...")
    gpu_ok, gpu_error = builder.check_gpu_available()
    if gpu_ok:
        print("   GPU available!")
    else:
        print(f"   GPU not available: {gpu_error}")


if __name__ == "__main__":
    main()
