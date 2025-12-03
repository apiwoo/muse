# Project MUSE - sam_wrapper.py
# Teacher: SAM 2 (Segment Anything 2) Video Predictor Wrapper
# (C) 2025 MUSE Corp. All rights reserved.

import os
import torch
import numpy as np
import cv2
import sys

# SAM 2 Imports
try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("❌ SAM 2 library not found. Please run 'pip install git+https://github.com/facebookresearch/segment-anything-2.git'")
    sys.exit(1)

class Sam2VideoWrapper:
    def __init__(self, model_root):
        """
        SAM 2 Video Predictor Wrapper
        - model_root: assets/models/segment_anything/
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"👨‍🏫 [Teacher A] SAM 2 (Hiera-Large) Initializing on {self.device}...")

        # 모델 경로 설정
        checkpoint = os.path.join(model_root, "sam2_hiera_large.pt")
        # SAM 2 Config는 라이브러리 내부에 있음 (sam2_hiera_l.yaml)
        model_cfg = "sam2_hiera_l.yaml"

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"❌ SAM 2 Checkpoint not found: {checkpoint}")

        try:
            self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=self.device)
            print("   ✅ SAM 2 Video Predictor Loaded.")
        except Exception as e:
            print(f"   ❌ SAM 2 Loading Failed: {e}")
            raise e
            
        self.inference_state = None

    def init_state(self, video_path):
        """비디오 세션 초기화 (전체 프레임 캐싱)"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        print(f"   🎞️ SAM 2: Initializing video state for {os.path.basename(video_path)}...")
        # SAM 2는 비디오 전체를 메모리에 올리거나 인덱싱함
        self.inference_state = self.predictor.init_state(video_path=video_path)

    def add_prompt(self, frame_idx, points=None, labels=None, box=None):
        """
        특정 프레임에 힌트(Prompt) 제공
        - points: [[x, y], ...]
        - labels: [1, 1, ...] (1: Foreground, 0: Background)
        """
        if self.inference_state is None:
            raise RuntimeError("Call init_state() first.")

        # SAM 2 API 호출
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
        비디오 전체에 마스크 전파 (Video Segmentation)
        Yields: (frame_idx, obj_ids, mask_logits)
        """
        if self.inference_state is None:
            raise RuntimeError("Call init_state() first.")
            
        print("   🌊 SAM 2: Propagating masks across video...")
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            yield out_frame_idx, out_obj_ids, out_mask_logits

    def reset(self):
        if self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
            self.inference_state = None