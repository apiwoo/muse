# Project MUSE - src/data_collection/guide_ui.py
# Created for AI Beauty Cam Project
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np

class GuideUI:
    def __init__(self):
        # 촬영 시나리오 (순서대로 진행)
        self.scenarios = [
            {"text": "정면을 바라보세요 (5초)", "duration": 5, "check": "frontal"},
            {"text": "천천히 고개를 왼쪽으로 90도 돌리세요", "duration": 5, "check": "left"},
            {"text": "천천히 고개를 오른쪽으로 90도 돌리세요", "duration": 5, "check": "right"},
            {"text": "위쪽을 바라보세요", "duration": 3, "check": "up"},
            {"text": "아래쪽을 바라보세요", "duration": 3, "check": "down"},
            {"text": "다양한 표정을 지어보세요 (아, 에, 이, 오, 우)", "duration": 10, "check": "expression"},
            {"text": "손으로 얼굴을 가렸다가 떼어보세요 (Occlusion)", "duration": 10, "check": "occlusion"},
        ]
        self.current_step = 0
        self.step_start_time = 0
        self.is_active = False

    def start(self):
        self.current_step = 0
        self.step_start_time = cv2.getTickCount()
        self.is_active = True

    def draw(self, frame):
        """프레임 위에 가이드 텍스트를 그립니다."""
        if not self.is_active or self.current_step >= len(self.scenarios):
            return

        current_scenario = self.scenarios[self.current_step]
        text = current_scenario["text"]
        duration = current_scenario["duration"]
        
        # 경과 시간 계산
        current_time = cv2.getTickCount()
        elapsed = (current_time - self.step_start_time) / cv2.getTickFrequency()
        remaining = max(0, duration - elapsed)
        
        # 다음 단계로 넘어감
        if remaining == 0:
            self.current_step += 1
            self.step_start_time = cv2.getTickCount()
            if self.current_step >= len(self.scenarios):
                self.is_active = False
                return

        # UI 그리기 (반투명 박스 + 텍스트)
        h, w = frame.shape[:2]
        
        # 박스 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 메인 텍스트
        cv2.putText(frame, text, (50, h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 타이머바 (진행률)
        progress = elapsed / duration
        bar_width = int(w * progress)
        cv2.rectangle(frame, (0, h - 10), (bar_width, h), (0, 255, 0), -1)