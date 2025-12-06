# Project MUSE - sam_wrapper.py
# Teacher: SAM 2.1 (Segment Anything 2.1) Video Predictor Wrapper
# (C) 2025 MUSE Corp. All rights reserved.

import os
import torch
import numpy as np
import cv2
import sys
import traceback

# [Fix] SAM 2 Imports with Hydra Handling
try:
    import sam2
    from sam2.build_sam import build_sam2_video_predictor
    import hydra
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
except ImportError:
    print("[ERROR] SAM 2 library not found. Please run 'pip install git+https://github.com/facebookresearch/segment-anything-2.git'")
    sys.exit(1)

class Sam2VideoWrapper:
    def __init__(self, model_root):
        """
        SAM 2.1 Video Predictor Wrapper
        - model_root: assets/models/segment_anything/
        - Updates: Upgraded to SAM 2.1 Large (Consistent with test_lightweight_sam.py)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TEACHER] [Teacher A] SAM 2.1 (Hiera-Large) Initializing on {self.device}...")

        # [Modified] Target SAM 2.1 Large
        checkpoint = os.path.join(model_root, "sam2.1_hiera_large.pt")
        model_cfg = "sam2.1_hiera_l.yaml"

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"[ERROR] SAM 2.1 Checkpoint not found: {checkpoint}\n   -> Run 'tools/download_models.py'")

        # [Hydra Fix] Reset existing config
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # [Core Logic] Find Config Dir
        config_dir = self._find_config_dir()
        if not config_dir:
            raise FileNotFoundError("[ERROR] SAM 2 Config directory not found.")

        print(f"   [DIR] Config Dir: {config_dir}")

        try:
            # Build model safely using Hydra Context Manager
            with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
                try:
                    # Attempt 1 (Standard) - Remove extension usually required by Hydra
                    cfg_name = model_cfg.replace(".yaml", "")
                    self.predictor = build_sam2_video_predictor(cfg_name, checkpoint, device=self.device)
                except Exception:
                    # Attempt 2 (Fallback with extension)
                    print("   [LOOP] Retrying with full config name...")
                    self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=self.device)
            
            print("   [OK] SAM 2.1 Video Predictor Loaded.")
            
        except Exception as e:
            print(f"   [ERROR] SAM 2.1 Loading Failed: {e}")
            traceback.print_exc()
            raise e
            
        self.inference_state = None

    def _find_config_dir(self):
        """
        Find directory containing config files.
        Priority 1: Project Local 'assets/sam2_configs/sam2.1' (Target Version)
        Priority 2: Installed Package
        """
        # 1. Project Local Asset (Deployment Mode)
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file)))))
        
        # [Modified] Look specifically for sam2.1 folder first
        local_config_v2_1 = os.path.join(project_root, "assets", "sam2_configs", "sam2.1")
        if os.path.exists(local_config_v2_1) and os.path.isdir(local_config_v2_1):
             print("   [INFO] Using Local Configs (SAM 2.1)")
             return local_config_v2_1

        # Fallback to general folder
        local_config = os.path.join(project_root, "assets", "sam2_configs")
        if os.path.exists(local_config) and os.path.isdir(local_config):
            if any(f.endswith(".yaml") for f in os.listdir(local_config)):
                return local_config

        # 2. Installed Package Search (Development Fallback)
        sam2_root = os.path.dirname(sam2.__file__)
        candidates = []
        
        for root, dirs, files in os.walk(sam2_root):
            # Look for 2.1 config first
            if "sam2.1_hiera_l.yaml" in files:
                candidates.insert(0, root) # Priority
            elif "sam2_hiera_l.yaml" in files:
                candidates.append(root)
        
        if candidates:
            # Prefer 'configs' dir if multiple found
            for d in candidates:
                if "configs" in d:
                    return d
            return candidates[0]
            
        return None

    def init_state(self, video_path):
        """Init Video Session"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        print(f"   [VIDEO] SAM 2.1: Initializing video state for {os.path.basename(video_path)}...")
        self.inference_state = self.predictor.init_state(video_path=video_path)

    def add_prompt(self, frame_idx, points=None, labels=None, box=None):
        """
        Provide Prompt to specific frame
        """
        if self.inference_state is None:
            raise RuntimeError("Call init_state() first.")

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=1, # Person ID
            points=points,
            labels=labels,
            box=box,
        )
        return out_mask_logits

    def propagate(self):
        """
        Propagate masks across the video
        Yields: (frame_idx, obj_ids, mask_logits)
        """
        if self.inference_state is None:
            raise RuntimeError("Call init_state() first.")
            
        print("   [FLOW] SAM 2.1: Propagating masks across video...")
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            yield out_frame_idx, out_obj_ids, out_mask_logits

    def reset(self):
        if self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
            self.inference_state = None