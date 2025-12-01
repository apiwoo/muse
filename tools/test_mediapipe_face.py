# Project MUSE - test_mediapipe_face.py
# MediaPipe Face Mesh vs InsightFace 비교 테스트용
# (C) 2025 MUSE Corp.

import cv2
import mediapipe as mp
import numpy as np
import time

class MediaPipeTester:
    def __init__(self):
        # 1. MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True: 눈동자(Iris) 좌표 포함 (478개 점)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 2. 카메라 설정
        self.cap = cv2.VideoCapture(0) # 0번 카메라 (안되면 1로 변경)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 3. 파라미터 (OpenCV 트랙바로 조절)
        self.params = {
            'eye_scale': 0, # 0 ~ 100
            'face_v': 0,    # 0 ~ 100
            'show_mesh': 1  # 0 or 1
        }
        
        # 윈도우 생성 및 트랙바 부착
        self.window_name = "MediaPipe Face Test"
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("Eye Size", self.window_name, 0, 100, lambda x: self._set_param('eye_scale', x))
        cv2.createTrackbar("V-Line", self.window_name, 0, 100, lambda x: self._set_param('face_v', x))
        cv2.createTrackbar("Show Mesh", self.window_name, 1, 1, lambda x: self._set_param('show_mesh', x))

    def _set_param(self, key, val):
        self.params[key] = val

    def run(self):
        print("🚀 MediaPipe 테스트 시작 (Press 'q' to quit)")
        prev_time = 0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            # BGR -> RGB 변환 (MediaPipe용)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            h, w, _ = frame.shape
            
            # 랜드마크가 감지되었을 때
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Normalized(0~1) -> Pixel(x,y) 변환
                    landmarks = np.array([
                        [int(pt.x * w), int(pt.y * h)] for pt in face_landmarks.landmark
                    ])
                    
                    # ---------------------------------------------------------
                    # [비교 포인트] MediaPipe 기반 성형 적용
                    # ---------------------------------------------------------
                    
                    # 1. 눈 키우기 (Eye Scaling)
                    if self.params['eye_scale'] > 0:
                        strength = self.params['eye_scale'] / 100.0
                        frame = self._warp_eyes_mp(frame, landmarks, strength)

                    # 2. 턱 깎기 (V-Line)
                    if self.params['face_v'] > 0:
                        strength = self.params['face_v'] / 100.0
                        frame = self._warp_face_mp(frame, landmarks, strength)

                    # 3. 디버그 메쉬 그리기
                    if self.params['show_mesh']:
                        self._draw_mesh(frame, landmarks)

            # FPS 계산
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            cv2.putText(frame, f"FPS: {int(fps)} (MediaPipe CPU)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    # ==========================================================================
    # [MediaPipe Specific Logic] 인덱스가 InsightFace와 다릅니다.
    # ==========================================================================
    def _warp_eyes_mp(self, img, lm, strength):
        """
        MediaPipe Iris Landmarks:
        - 468: Left Iris Center
        - 473: Right Iris Center
        """
        # 왼쪽 눈
        center_l = lm[468] 
        # 눈 너비 추정 (좌우 끝점: 33, 133)
        eye_width_l = np.linalg.norm(lm[33] - lm[133])
        radius_l = int(eye_width_l * 1.5) # InsightFace보다 범위 약간 좁게 잡음

        # 오른쪽 눈
        center_r = lm[473]
        # 눈 너비 추정 (좌우 끝점: 362, 263)
        eye_width_r = np.linalg.norm(lm[362] - lm[263])
        radius_r = int(eye_width_r * 1.5)

        img = self._apply_local_warp(img, center_l, radius_l, strength, mode='expand')
        img = self._apply_local_warp(img, center_r, radius_r, strength, mode='expand')
        return img

    def _warp_face_mp(self, img, lm, strength):
        """
        MediaPipe Face Contour Indices (V-Line 수정됨):
        - 기존 상단(귀 근처) 좌표 제거 -> 하관 집중
        - 턱 끝: 152
        """
        chin_pt = lm[152] # 목표점 (턱 끝)

        # [수정됨] 왼쪽 턱 당기기 (관자놀이 제외, 턱선 집중)
        # 기존: [234, 93, 132, 58, 172] -> 234, 93(귀/광대) 제거
        # 신규: [132, 58, 172, 136, 150] -> 귀 밑 사각턱부터 턱 끝 라인
        left_indices = [132, 58, 172, 136, 150]
        
        for idx in left_indices:
            pt = lm[idx]
            dist = np.linalg.norm(pt - chin_pt)
            radius = int(dist * 0.6) # 반경을 조금 키워서 부드럽게
            vector = chin_pt - pt 
            img = self._apply_local_warp(img, pt, radius, strength * 0.3, mode='shrink', vector=vector)

        # [수정됨] 오른쪽 턱 당기기
        # 기존: [454, 323, 361, 288, 397] -> 454, 323(귀/광대) 제거
        # 신규: [361, 288, 397, 365, 379] -> 귀 밑 사각턱부터 턱 끝 라인
        right_indices = [361, 288, 397, 365, 379]
        
        for idx in right_indices:
            pt = lm[idx]
            dist = np.linalg.norm(pt - chin_pt)
            radius = int(dist * 0.6)
            vector = chin_pt - pt
            img = self._apply_local_warp(img, pt, radius, strength * 0.3, mode='shrink', vector=vector)

        return img

    def _draw_mesh(self, img, lm):
        # 촘촘한 점들 그려서 안정성 확인 (초록색)
        for p in lm:
            cv2.circle(img, tuple(p), 1, (0, 255, 0), -1)
        
        # 주요 부위 강조 (빨간색: 눈동자, 턱끝)
        cv2.circle(img, tuple(lm[468]), 3, (0, 0, 255), -1) # 좌안 중심
        cv2.circle(img, tuple(lm[473]), 3, (0, 0, 255), -1) # 우안 중심
        cv2.circle(img, tuple(lm[152]), 3, (0, 0, 255), -1) # 턱 끝
        
        # [Debug] V-Line 적용 포인트 확인용 (노란색)
        # 왼쪽
        for idx in [132, 58, 172, 136, 150]:
             cv2.circle(img, tuple(lm[idx]), 3, (0, 255, 255), -1)
        # 오른쪽
        for idx in [361, 288, 397, 365, 379]:
             cv2.circle(img, tuple(lm[idx]), 3, (0, 255, 255), -1)

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
    tester = MediaPipeTester()
    tester.run()