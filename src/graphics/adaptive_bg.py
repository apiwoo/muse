# Project MUSE - adaptive_bg.py
# V6: MODNet + FaceMesh + ViTPose + 배경유사도 4중 검증 배경 업데이트
# (C) 2025 MUSE Corp. All rights reserved.

import cupy as cp
import cupyx.scipy.ndimage
import cv2
import numpy as np
import os
import threading

class AdaptiveBackground:
    def __init__(self, width=1920, height=1080):
        self.w = width
        self.h = height
        self.bg_buffer = None  # Float32
        self.original_static_bg = None  # [V8] 원본 정적 배경 (오염 방지용)
        self.last_frame = None  # [V6] 마지막 프레임 저장 (검은화면 방지용)

        self.is_static_loaded = False

        # [V5.0] 파일 자동 저장 설정
        self.bg_file_path = None
        self.update_counter = 0
        self.file_save_interval = 600  # 약 20초마다 파일 저장 (30fps 기준)

        # [V5.2] 연속 프레임 검증 + 공간 마진
        self.temporal_counter = None  # (H, W) 각 픽셀의 연속 배경 프레임 수
        self.TEMPORAL_THRESHOLD = 15  # 15프레임 연속 배경이어야 업데이트 (0.5초)
        self.ALPHA_THRESHOLD = 0.005  # alpha < 0.005 = 확실한 배경
        self.SPATIAL_SIZE = 15        # [V8] 15x15 이웃 검사 (안전 마진 확대)

        # [V7] Historical Person Mask 누적 시스템
        self.HISTORY_FRAMES = 60      # 최근 60프레임 히스토리 (약 2초)
        self.person_mask_history = None  # 순환 버퍼, shape: (HISTORY_FRAMES, H, W)
        self.history_index = 0        # 순환 버퍼 현재 인덱스

        # [V6] 랜드마크 기반 사람 판단 설정
        self.LANDMARK_RADIUS_FACE = 15   # FaceMesh 포인트 주변 반경 (픽셀)
        self.LANDMARK_RADIUS_BODY = 40   # ViTPose 포인트 주변 반경 (픽셀)
        self.BODY_CONF_THRESHOLD = 0.3   # ViTPose 신뢰도 임계값

        # FaceMesh FACE_OVAL 인덱스 (36개 포인트로 얼굴 외곽 정의)
        self.FACE_OVAL_INDICES = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]

        # [V6] 배경 유사도 검사 설정
        self.BG_SIMILARITY_THRESHOLD = 20.0  # [V8] 원본 기준 비교로 임계값 하향 (25→20)
        # 값이 작을수록 엄격 (배경 업데이트 어려움)
        # 값이 클수록 느슨 (배경 업데이트 쉬움)
        # 권장: 20~30 (조명 변화 허용하면서 사람/물체 구분)

        # [V9] 프레임 간 움직임 감지 설정
        self.MOTION_THRESHOLD = 5.0  # 이전 프레임과의 픽셀 차이 허용치
        self.prev_frame_for_motion = None  # 이전 프레임 저장용

    def load_static_background(self, bg_source):
        frame_bgr = None
        if isinstance(bg_source, str):
            if os.path.exists(bg_source):
                frame_bgr = cv2.imread(bg_source)
                self.bg_file_path = bg_source  # [V5.0] 파일 경로 저장
                print(f"[BG] Loaded static background from: {bg_source}")
            else:
                print(f"[BG] Background file not found: {bg_source}")
                return False
        elif isinstance(bg_source, np.ndarray):
            frame_bgr = bg_source

        if frame_bgr is not None:
            if frame_bgr.shape[1] != self.w or frame_bgr.shape[0] != self.h:
                frame_bgr = cv2.resize(frame_bgr, (self.w, self.h))
            self.bg_buffer = cp.asarray(frame_bgr).astype(cp.float32)
            self.original_static_bg = self.bg_buffer.copy()  # [V8] 원본 백업
            self.is_static_loaded = True
            self.temporal_counter = None  # [V5.1] 리셋
            self.prev_frame_for_motion = None  # [V9] 움직임 감지용 리셋
            # [V7] 히스토리 버퍼 리셋
            self.person_mask_history = None
            self.history_index = 0
            print("[BG] Static Background Loaded. Real-time update enabled for BG areas.")
            return True
        return False

    def set_file_path(self, path):
        """[V5.0] 배경 파일 경로 설정"""
        self.bg_file_path = path

    def reset(self, frame_gpu):
        if frame_gpu is None: return

        # [V6] 항상 마지막 프레임 저장 (검은화면 방지)
        self.last_frame = frame_gpu.astype(cp.float32)

        # 정적 배경이 로드되어 있다면 리셋하지 않음 (보호)
        if self.is_static_loaded: return

        self.bg_buffer = frame_gpu.astype(cp.float32)
        # [V7] 히스토리 버퍼 리셋
        self.person_mask_history = None
        self.history_index = 0
        # [V8] 원본 정적 배경 리셋
        self.original_static_bg = None
        # [V9] 움직임 감지용 이전 프레임 리셋
        self.prev_frame_for_motion = None
        print("[BG] Background Buffer Reset (Initialized with Live Frame)")

    def _generate_landmark_mask(self, h, w, faces, keypoints):
        """
        [V6] FaceMesh/ViTPose 좌표 기반 사람 영역 마스크 생성

        원칙: 랜드마크가 검출된 좌표 주변은 반드시 사람 영역
        - FaceMesh: 얼굴 영역 보호
        - ViTPose: 신체 영역 보호

        Returns:
            CuPy array (h, w), 값 0.0(배경) 또는 1.0(사람)
        """
        # CPU에서 마스크 생성 (OpenCV 사용)
        mask_cpu = np.zeros((h, w), dtype=np.float32)

        # 1. FaceMesh 처리
        if faces is not None and len(faces) > 0:
            for face in faces:
                if not hasattr(face, 'landmarks'):
                    continue
                landmarks = face.landmarks  # (478, 2)

                # 1-1. 각 랜드마크 주변 원형 영역
                for pt in landmarks:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(mask_cpu, (x, y), self.LANDMARK_RADIUS_FACE, 1.0, -1)

                # 1-2. FACE_OVAL로 얼굴 전체 영역 채우기
                if len(landmarks) >= 468:  # 전체 랜드마크가 있는 경우만
                    oval_pts = landmarks[self.FACE_OVAL_INDICES].astype(np.int32)
                    cv2.fillConvexPoly(mask_cpu, oval_pts, 1.0)

        # 2. ViTPose 처리
        if keypoints is not None and len(keypoints) >= 13:
            # 2-1. 각 키포인트 주변 원형 영역
            for kp in keypoints:
                x, y, conf = kp[0], kp[1], kp[2]
                if conf > self.BODY_CONF_THRESHOLD:
                    ix, iy = int(x), int(y)
                    if 0 <= ix < w and 0 <= iy < h:
                        cv2.circle(mask_cpu, (ix, iy), self.LANDMARK_RADIUS_BODY, 1.0, -1)

            # 2-2. 몸통 영역 (어깨-골반) convex hull로 채우기
            # COCO 인덱스: 5=L_Shoulder, 6=R_Shoulder, 11=L_Hip, 12=R_Hip
            torso_indices = [5, 6, 12, 11]  # 시계방향 순서
            torso_pts = []
            for idx in torso_indices:
                if idx < len(keypoints) and keypoints[idx][2] > self.BODY_CONF_THRESHOLD:
                    torso_pts.append([int(keypoints[idx][0]), int(keypoints[idx][1])])

            if len(torso_pts) >= 3:  # 최소 3점 이상이어야 폴리곤 가능
                torso_pts = np.array(torso_pts, dtype=np.int32)
                cv2.fillConvexPoly(mask_cpu, torso_pts, 1.0)

        # GPU로 업로드
        return cp.asarray(mask_cpu)

    def _compute_bg_similarity_mask(self, frame_gpu):
        """
        [V8] 배경 유사도 기반 마스크 생성 (원본 정적 배경 기준)

        원칙: 원본 정적 배경과 현재 프레임이 다르면 무언가 있다는 의미
        - 유사함 (차이 < 임계값): 배경 업데이트 허용
        - 다름 (차이 >= 임계값): 배경 업데이트 불허

        [V8 변경] bg_buffer 대신 original_static_bg와 비교하여 오염 악순환 방지

        Args:
            frame_gpu: 현재 프레임 (CuPy, uint8 또는 float32)

        Returns:
            CuPy array (h, w), True=유사(배경), False=다름(사람/물체)
        """
        if self.original_static_bg is None:
            # 원본 정적 배경이 없으면 모두 유사하다고 판단 (업데이트 허용)
            h, w = frame_gpu.shape[:2]
            return cp.ones((h, w), dtype=cp.bool_)

        # 현재 프레임을 float32로 변환
        current_float = frame_gpu.astype(cp.float32)

        # 픽셀별 색상 차이 계산 (L1 거리, 채널 평균)
        # [V8] 원본 정적 배경과 비교 (bg_buffer가 아닌 original_static_bg)
        diff = cp.abs(current_float - self.original_static_bg)  # (H, W, 3)
        diff_mean = cp.mean(diff, axis=2)  # (H, W) 채널 평균

        # 유사도 판정: 차이가 임계값 미만이면 유사
        is_similar = (diff_mean < self.BG_SIMILARITY_THRESHOLD)

        return is_similar

    def _compute_motion_mask(self, frame_gpu):
        """
        [V9] 프레임 간 움직임 감지 마스크 생성

        원칙: 이전 프레임과 현재 프레임이 동일해야 배경 업데이트 허용
        - 정지 상태 (차이 < 임계값): True (업데이트 허용)
        - 움직임 있음 (차이 >= 임계값): False (업데이트 금지)

        Args:
            frame_gpu: 현재 프레임 (CuPy, uint8 또는 float32)

        Returns:
            CuPy array (h, w), True=정지(업데이트 허용), False=움직임(업데이트 금지)
        """
        h, w = frame_gpu.shape[:2]
        current_float = frame_gpu.astype(cp.float32)

        if self.prev_frame_for_motion is None:
            self.prev_frame_for_motion = current_float.copy()
            return cp.ones((h, w), dtype=cp.bool_)

        # 픽셀별 색상 차이 계산 (L1 거리, 채널 평균)
        diff = cp.abs(current_float - self.prev_frame_for_motion)
        diff_mean = cp.mean(diff, axis=2)

        # 정지 상태 판정: 차이가 임계값 미만이면 정지
        is_static = (diff_mean < self.MOTION_THRESHOLD)

        # 현재 프레임을 다음 비교를 위해 저장
        self.prev_frame_for_motion = current_float.copy()

        return is_static

    def update(self, frame_gpu, person_alpha, faces=None, keypoints=None):
        """
        [V6] 배경 업데이트 (4중 검증)

        검증 1: MODNet - person_alpha가 임계값 미만
        검증 2: FaceMesh - 얼굴 랜드마크 주변 아님
        검증 3: ViTPose - 신체 키포인트 주변 아님
        검증 4: 배경유사도 - 기존 배경과 색상이 유사함

        4가지 모두 통과해야 배경 업데이트 허용

        Args:
            frame_gpu: 현재 프레임 (CuPy)
            person_alpha: MODNet 사람 마스크 (CuPy, 0.0~1.0)
            faces: FaceMesh 결과 리스트 (각 요소는 .landmarks 속성)
            keypoints: ViTPose 결과 (17, 3) [x, y, conf]
        """
        if self.bg_buffer is None or frame_gpu is None:
            if frame_gpu is not None:
                self.reset(frame_gpu)
            return

        # [V6] 항상 마지막 프레임 저장 (검은화면 방지)
        self.last_frame = frame_gpu.astype(cp.float32)

        # [V10] 정적 배경 모드: 업데이트 완전 비활성화
        # 조명 적응 기능 필요시 아래 if 블록을 주석처리하고 기존 로직 활성화
        if self.is_static_loaded:
            return

        # ============================================================
        # [비활성화] 아래 모든 업데이트 로직은 정적 배경 모드에서 실행되지 않음
        # 정적 배경이 없는 경우에만 아래 로직 실행
        # ============================================================

        h, w = frame_gpu.shape[:2]

        # 1. temporal_counter 초기화
        if self.temporal_counter is None or self.temporal_counter.shape != (h, w):
            self.temporal_counter = cp.zeros((h, w), dtype=cp.uint8)

        # 1-1. [V7] person_mask_history 초기화
        if self.person_mask_history is None or self.person_mask_history.shape[1:] != (h, w):
            self.person_mask_history = cp.zeros((self.HISTORY_FRAMES, h, w), dtype=cp.uint8)
            self.history_index = 0

        # 2. 알파 마스크 2D 변환
        alpha_2d = person_alpha
        if alpha_2d.ndim == 3:
            alpha_2d = alpha_2d.squeeze()

        # ================================================================
        # [검증 1] MODNet 기반 배경 판정
        # ================================================================
        # 3. 공간 검증: 7x7 이웃 중 최대 alpha 계산
        neighbor_max_alpha = cupyx.scipy.ndimage.maximum_filter(
            alpha_2d, size=self.SPATIAL_SIZE
        )
        is_bg_by_modnet = (neighbor_max_alpha < self.ALPHA_THRESHOLD)

        # ================================================================
        # [V7] Historical Person Mask 누적 시스템
        # ================================================================
        # Step A: 현재 프레임의 사람 영역을 이진 마스크로 변환
        # 임계값 0.1은 경계부까지 포함하기 위해 ALPHA_THRESHOLD(0.005)보다 높게 설정
        current_person_mask = (alpha_2d >= 0.1).astype(cp.uint8)

        # Step B: 순환 버퍼에 현재 마스크 저장
        self.person_mask_history[self.history_index] = current_person_mask
        self.history_index = (self.history_index + 1) % self.HISTORY_FRAMES

        # Step C: 누적 마스크 계산 (최근 60프레임 중 한 번이라도 사람이었던 픽셀)
        accumulated_person_mask = cp.any(self.person_mask_history, axis=0)

        # ================================================================
        # [검증 2+3] 랜드마크 기반 사람 영역 판정
        # ================================================================
        landmark_mask = self._generate_landmark_mask(h, w, faces, keypoints)
        is_person_by_landmark = (landmark_mask > 0.5)
        is_bg_by_landmark = ~is_person_by_landmark

        # ================================================================
        # [검증 4] 배경 유사도 기반 판정
        # ================================================================
        is_similar_to_bg = self._compute_bg_similarity_mask(frame_gpu)

        # ================================================================
        # [검증 5] 프레임 간 움직임 감지 (V9)
        # ================================================================
        is_static = self._compute_motion_mask(frame_gpu)

        # ================================================================
        # [최종 판정] 5중 검증 AND 연산
        # ================================================================
        is_bg_now = is_bg_by_modnet & is_bg_by_landmark & is_similar_to_bg & is_static

        # ================================================================
        # [V7] Historical Person Mask 조건 적용
        # ================================================================
        # 최근 60프레임 중 한 번이라도 사람이었던 영역은 배경 업데이트 불허
        is_person_recently = accumulated_person_mask
        is_truly_safe = is_bg_now & (~is_person_recently)

        # 7. temporal_counter 업데이트
        # 안전한 픽셀: counter += 1 (최대 255)
        # 위험한 픽셀 (최근 사람 있었음): counter = 0 (즉시 리셋)
        self.temporal_counter = cp.where(
            is_truly_safe,
            cp.minimum(self.temporal_counter + 1, 255).astype(cp.uint8),
            cp.uint8(0)
        )

        # 8. N프레임 연속 배경인 픽셀만 업데이트
        is_safe_to_update = (self.temporal_counter >= self.TEMPORAL_THRESHOLD)
        update_mask = is_safe_to_update[..., None]  # (H, W, 1)

        # 9. 배경 업데이트 (안전한 영역만)
        current_float = frame_gpu.astype(cp.float32)
        self.bg_buffer = cp.where(update_mask, current_float, self.bg_buffer)

        # 10. 주기적 파일 저장
        if self.bg_file_path:
            self.update_counter += 1
            if self.update_counter >= self.file_save_interval:
                self.update_counter = 0
                self._save_background_async()

    def _save_background_async(self):
        """[V5.0] 배경 버퍼를 파일에 비동기 저장"""
        if self.bg_buffer is None or self.bg_file_path is None:
            return

        try:
            bg_copy = self.bg_buffer.get().astype(np.uint8)
        except Exception as e:
            print(f"[BG] Failed to copy buffer: {e}")
            return

        def save_worker(img, path):
            try:
                cv2.imwrite(path, img)
                print(f"[BG] Auto-saved background to: {path}")
            except Exception as e:
                print(f"[BG] Failed to save: {e}")

        t = threading.Thread(target=save_worker, args=(bg_copy, self.bg_file_path))
        t.daemon = True
        t.start()

    def is_background_stable(self):
        """
        [V6] 배경 안정성 확인
        정적 배경이 로드된 경우에만 True 반환.
        - True: 파일에서 로드되었거나 'B'키로 캡처됨 → 슬리밍 합성 가능
        - False: 적응형 배경 모드 → 슬리밍 합성 시 일렁임 발생 가능
        """
        return self.is_static_loaded

    def get_background(self):
        if self.bg_buffer is None:
            # [V6] 마지막 프레임으로 fallback (검은화면 방지)
            if self.last_frame is not None:
                return self.last_frame.astype(cp.uint8)
            return cp.zeros((self.h, self.w, 3), dtype=cp.uint8)
        return self.bg_buffer.astype(cp.uint8)