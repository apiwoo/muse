# Project MUSE - test_mediapipe_pose.py
# MediaPipe Pose vs ViTPose(TensorRT) 비교 테스트용
# (C) 2025 MUSE Corp.

import cv2
import mediapipe as mp
import numpy as np
import time

class MediaPipePoseTester:
    def __init__(self):
        # 1. MediaPipe Pose 초기화
        self.mp_pose = mp.solutions.pose
        # min_detection_confidence: 감지 임계값
        # min_tracking_confidence: 추적 임계값 (높을수록 덜 튀지만 놓칠 확률 있음)
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # 0: Lite, 1: Full, 2: Heavy
        )
        
        # 2. 카메라 설정
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 3. 파라미터 (OpenCV 트랙바로 조절)
        self.params = {
            'waist_slim': 0,    # 0 ~ 100
            'show_skeleton': 1  # 0 or 1
        }
        
        # 윈도우 생성 및 트랙바 부착
        self.window_name = "MediaPipe Pose Test"
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("Waist Slim", self.window_name, 0, 100, lambda x: self._set_param('waist_slim', x))
        cv2.createTrackbar("Show Skeleton", self.window_name, 1, 1, lambda x: self._set_param('show_skeleton', x))

    def _set_param(self, key, val):
        self.params[key] = val

    def run(self):
        print("🚀 MediaPipe Pose 테스트 시작 (Press 'q' to quit)")
        prev_time = 0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            h, w, _ = frame.shape
            
            # BGR -> RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 추론 수행
            results = self.pose.process(frame_rgb)

            # 랜드마크가 감지되었을 때
            if results.pose_landmarks:
                # Normalized(0~1) -> Pixel(x,y) 변환
                landmarks = np.array([
                    [int(pt.x * w), int(pt.y * h)] for pt in results.pose_landmarks.landmark
                ])
                
                # ---------------------------------------------------------
                # [비교 포인트] MediaPipe 기반 허리 성형
                # ---------------------------------------------------------
                if self.params['waist_slim'] > 0:
                    strength = self.params['waist_slim'] / 100.0
                    frame = self._warp_waist_mp(frame, landmarks, strength)

                # 뼈대 그리기
                if self.params['show_skeleton']:
                    self._draw_skeleton(frame, landmarks, results.pose_landmarks)

            # FPS 계산
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            cv2.putText(frame, f"FPS: {int(fps)} (MediaPipe Pose)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    # ==========================================================================
    # [MediaPipe Specific Logic] 인덱스가 ViTPose(COCO)와 다릅니다.
    # ==========================================================================
    def _warp_waist_mp(self, img, lm, strength):
        """
        MediaPipe Pose Landmarks (BlazePose):
        - 11: Left Shoulder
        - 12: Right Shoulder
        - 23: Left Hip
        - 24: Right Hip
        """
        # 좌표 추출 (이미 픽셀 단위)
        try:
            l_sh = lm[11]
            r_sh = lm[12]
            l_hip = lm[23]
            r_hip = lm[24]
        except IndexError:
            return img

        # 허리 위치 추정 (어깨와 힙의 0.6 지점 - 힙 쪽에 더 가깝게)
        l_waist = l_sh * 0.4 + l_hip * 0.6
        r_waist = r_sh * 0.4 + r_hip * 0.6
        
        # 몸통 중심선
        center_waist = (l_waist + r_waist) / 2
        
        # 워핑 반경 (몸통 너비 기반)
        body_width = np.linalg.norm(l_waist - r_waist)
        if body_width < 10: return img # 너무 작거나 옆모습이면 패스
        
        radius = int(body_width * 0.6)
        
        # 강도 조절
        warp_strength = strength * 0.4

        # 왼쪽 허리 당기기 (중심 쪽으로)
        vec_l = center_waist - l_waist
        img = self._apply_local_warp(img, l_waist, radius, warp_strength, mode='shrink', vector=vec_l)

        # 오른쪽 허리 당기기 (중심 쪽으로)
        vec_r = center_waist - r_waist
        img = self._apply_local_warp(img, r_waist, radius, warp_strength, mode='shrink', vector=vec_r)

        return img

    def _draw_skeleton(self, img, lm, raw_landmarks):
        # MediaPipe 유틸리티로 그리면 예쁘지만, 비교를 위해 단순하게 그립니다.
        mp.solutions.drawing_utils.draw_landmarks(
            img,
            raw_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
        )
        
        # 주요 부위 강조 (어깨, 힙)
        indices = [11, 12, 23, 24]
        for idx in indices:
            cv2.circle(img, tuple(lm[idx]), 5, (0, 0, 255), -1)

    # ==========================================================================
    # [Warping Algorithm] beauty_engine.py 에서 복사 (의존성 제거)
    # ==========================================================================
    def _apply_local_warp(self, img, center, radius, strength, mode='expand', vector=None):
        cx, cy = int(center[0]), int(center[1])
        r = int(radius)
        
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(img.shape[1], cx + r), min(img.shape[0], cy + r)
        
        roi = img[y1:y2, x1:x2]
        if roi.size == 0: return img

        h, w = roi.shape[:2]
        grid_y, grid_x = np.indices((h, w), dtype=np.float32)
        
        lcx, lcy = cx - x1, cy - y1
        dx = grid_x - lcx
        dy = grid_y - lcy
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq)
        
        mask = dist < r
        factor = np.zeros_like(dist)
        with np.errstate(divide='ignore', invalid='ignore'):
             factor[mask] = (1.0 - dist[mask] / r) ** 2 * strength

        map_x = grid_x.copy()
        map_y = grid_y.copy()

        if mode == 'expand':
            map_x[mask] -= dx[mask] * factor[mask]
            map_y[mask] -= dy[mask] * factor[mask]
            
        elif mode == 'shrink':
            if vector is not None:
                vx, vy = vector
                v_len = np.sqrt(vx*vx + vy*vy)
                if v_len > 0:
                    vx, vy = vx/v_len, vy/v_len
                    map_x[mask] -= vx * factor[mask] * r * 0.5
                    map_y[mask] -= vy * factor[mask] * r * 0.5
            else:
                map_x[mask] += dx[mask] * factor[mask]
                map_y[mask] += dy[mask] * factor[mask]

        warped_roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
        
        mask_img = np.zeros((h, w), dtype=np.float32)
        mask_img[mask] = 1.0
        mask_img = cv2.GaussianBlur(mask_img, (5, 5), 0)
        
        mask_3ch = mask_img[..., np.newaxis]
        img[y1:y2, x1:x2] = (warped_roi * mask_3ch + roi * (1.0 - mask_3ch)).astype(np.uint8)
        
        return img

if __name__ == "__main__":
    tester = MediaPipePoseTester()
    tester.run()