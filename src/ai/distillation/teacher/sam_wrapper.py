# Project MUSE - sam_wrapper.py
# Teacher: SAM 2 (Segment Anything 2) Video Predictor Wrapper
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
        model_cfg = "sam2_hiera_l.yaml"

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"❌ SAM 2 Checkpoint not found: {checkpoint}")

        # [Hydra Fix] 기존 설정 초기화
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # [Core Logic] Config 경로 찾기 (우선순위 적용)
        config_dir = self._find_config_dir()
        if not config_dir:
            raise FileNotFoundError("❌ SAM 2 Config directory not found.")

        print(f"   📂 Config Dir: {config_dir}")

        try:
            # Hydra Context Manager를 사용하여 안전하게 모델 빌드
            with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
                try:
                    # 1차 시도 (확장자 포함)
                    self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=self.device)
                except Exception:
                    # 2차 시도 (확장자 제거 - Hydra 표준)
                    print("   🔄 Retrying with config name adjustment...")
                    cfg_name = model_cfg.replace(".yaml", "")
                    self.predictor = build_sam2_video_predictor(cfg_name, checkpoint, device=self.device)
            
            print("   ✅ SAM 2 Video Predictor Loaded.")
            
        except Exception as e:
            print(f"   ❌ SAM 2 Loading Failed: {e}")
            traceback.print_exc()
            raise e
            
        self.inference_state = None

    def _find_config_dir(self):
        """
        설정 파일이 있는 폴더를 찾습니다.
        Priority 1: 프로젝트 내부 'assets/sam2_configs' (배포용/안정적)
        Priority 2: 설치된 패키지 내부 (개발용)
        """
        # 1. Project Local Asset (Recommended for Deployment)
        # 현재 파일: src/ai/distillation/teacher/sam_wrapper.py
        # 루트: src/../../.. -> Project Root
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file)))))
        local_config = os.path.join(project_root, "assets", "sam2_configs")
        
        if os.path.exists(local_config) and os.path.isdir(local_config):
            # yaml 파일이 실제로 있는지 확인
            if any(f.endswith(".yaml") for f in os.listdir(local_config)):
                print("   ✨ Using Local Configs (Deployment Mode)")
                return local_config

        # 2. Installed Package Search (Development Fallback)
        sam2_root = os.path.dirname(sam2.__file__)
        candidates = []
        
        # 패키지 내부 검색
        for root, dirs, files in os.walk(sam2_root):
            if "sam2_hiera_l.yaml" in files:
                candidates.append(root)
        
        if candidates:
            # 'configs'가 경로명에 포함된 것을 선호
            for d in candidates:
                if "configs" in d:
                    return d
            return candidates[0]
            
        return None

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