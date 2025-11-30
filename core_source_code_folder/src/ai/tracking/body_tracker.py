# Project MUSE - body_tracker.py
# Target: RTX 3060/4090 Mode A (High Performance)
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
from ai.tracking.vitpose_trt import VitPoseTrt

class BodyTracker:
    def __init__(self):
        """
        [Body Tracking Engine]
        - Engine: ViTPose-Huge (TensorRT Accelerated)
        - Keypoints: COCO 17 Format
        """
        print("💪 [BodyTracker] ViTPose-Huge 엔진(TensorRT) 로드 중...")
        try:
            self.model = VitPoseTrt()
            self.engine_ready = True
        except Exception as e:
            print(f"❌ [BodyTracker] 엔진 로드 실패: {e}")
            print("   (tools/trt_converter.py가 성공적으로 실행되었는지 확인하세요)")
            self.engine_ready = False

    def process(self, frame_bgr):
        """
        :return: keypoints numpy array (17, 3) -> [x, y, conf]
        """
        if not self.engine_ready or frame_bgr is None:
            return None
        
        # ViTPose 추론 실행
        # (전처리, GPU 추론, 후처리가 내부에서 최적화됨)
        keypoints = self.model.inference(frame_bgr)
        
        return keypoints

    def draw_debug(self, frame, keypoints):
        """
        [Visual Check] COCO 포맷(17 Keypoints) 뼈대 그리기
        """
        if keypoints is None:
            return frame

        # COCO Keypoint Index:
        # 0:코, 1:왼눈, 2:오른눈, 3:왼귀, 4:오른귀
        # 5:왼어깨, 6:오른어깨, 7:왼팔꿈치, 8:오른팔꿈치, 9:왼손목, 10:오른손목
        # 11:왼골반, 12:오른골반, 13:왼무릎, 14:오른무릎, 15:왼발목, 16:오른발목

        # 신뢰도 임계값 (이 값보다 낮으면 안 그림)
        CONF_THRESH = 0.3

        # 1. 점 찍기
        for i in range(17):
            x, y, conf = keypoints[i]
            if conf > CONF_THRESH:
                # 관절마다 색깔 다르게 (좌:파랑, 우:빨강)
                color = (255, 100, 0) if i % 2 == 1 else (0, 100, 255)
                if i == 0: color = (0, 255, 255) # 코는 노란색
                
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)

        # 2. 선 연결 (Skeleton)
        skeleton = [
            (5, 7), (7, 9),       # 왼팔
            (6, 8), (8, 10),      # 오른팔
            (11, 13), (13, 15),   # 왼다리
            (12, 14), (14, 16),   # 오른다리
            (5, 6),               # 어깨선
            (11, 12),             # 골반선
            (5, 11), (6, 12),     # 몸통 (어깨-골반)
            (0, 1), (0, 2),       # 얼굴 (코-눈)
            (1, 3), (2, 4)        # 얼굴 (눈-귀)
        ]

        for p1, p2 in skeleton:
            x1, y1, c1 = keypoints[p1]
            x2, y2, c2 = keypoints[p2]
            
            if c1 > CONF_THRESH and c2 > CONF_THRESH:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        return frame

    # [Compatibility] MediaPipe 포맷 지원용 (BeautyEngine 호환성 유지)
    # ViTPose 결과를 MediaPipe Landmark 객체처럼 포장해서 리턴할 수도 있지만,
    # 지금은 BeautyEngine을 수정하는 편이 더 깔끔하므로 여기선 Raw Data를 넘깁니다.
    # 하지만 BeautyEngine은 아직 MediaPipe 포맷을 기대하고 있으므로
    # 다음 단계에서 BeautyEngine의 _warp_waist 함수도 수정해야 합니다.