# Project MUSE - facemesh.py
# Target: MediaPipe Migration (Lightweight & High Detail)
# Updated V25.0: Extended Indices for Polygon Masking & Frequency Separation
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import mediapipe as mp

class FaceMesh:
    # [MediaPipe Indices Map]
    # 성형 엔진에서 사용할 부위별 랜드마크 인덱스 매핑 (478 Landmarks 기준)
    FACE_INDICES = {
        # ============================================================
        # [기존 인덱스 - 100% 유지]
        # ============================================================

        # [눈] (왼쪽/오른쪽) - 눈꺼풀 위아래 및 눈꼬리 포함
        "EYE_L": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 144, 145, 153],
        "EYE_R": [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390],

        # [눈썹] (Eyebrows) - 보호 구역 설정을 위해 추가
        "BROW_L": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        "BROW_R": [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],

        # [입술] (Lips) - 보호 구역 설정을 위해 추가 (외곽선 위주)
        "LIPS": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 267, 185, 40, 39, 37, 0],

        # [턱 라인] (V라인 성형용)
        "JAW_L": [
            234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152
        ],
        "JAW_R": [
            454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152
        ],

        # [코] (중심점 및 콧볼)
        "NOSE_TIP": [1],
        "NOSE_WING_L": [279],
        "NOSE_WING_R": [49],

        # ============================================================
        # [V25.0 신규 인덱스 - 폴리곤 마스킹용]
        # ============================================================

        # [얼굴 전체 윤곽] - Convex Hull 및 폴리곤 마스크 생성용
        # 이마 상단부터 시계방향으로 정렬된 36개 포인트
        "FACE_OVAL": [
            10,   # 이마 중앙 상단
            338, 297, 332, 284,  # 이마 우측
            251, 389, 356,       # 우측 관자놀이
            454, 323, 361, 288,  # 우측 볼
            397, 365, 379, 378, 400, 377,  # 우측 턱
            152,  # 턱 끝 (중앙)
            148, 176, 149, 150, 136, 172, 58, 132, 93, 234,  # 좌측 턱 ~ 볼
            127, 162, 21,        # 좌측 관자놀이
            54, 103, 67, 109     # 이마 좌측
        ],

        # [이마 영역] - 피부 스무딩 확장용 (눈썹 위)
        "FOREHEAD": [
            10, 338, 297, 332, 284, 251,  # 이마 우측 라인
            21, 54, 103, 67, 109,         # 이마 좌측 라인
            108, 69, 104, 68, 71,         # 이마 내부 (눈썹 바로 위)
            336, 296, 334, 293, 300,      # 우측 눈썹 위
            70, 63, 105, 66, 107          # 좌측 눈썹 위
        ],

        # [볼 영역] - 피부 스무딩 핵심 영역
        "CHEEK_L": [
            36, 142, 126, 217, 174, 196, 197, 419, 399, 437
        ],
        "CHEEK_R": [
            266, 371, 355, 437, 399, 419, 197, 196, 174, 217
        ],

        # [코 전체] - 코 영역 마스킹용
        "NOSE_FULL": [
            168, 6, 197, 195, 5,   # 코 중앙선
            4, 1,                   # 코끝
            279, 49,                # 콧볼 좌우
            274, 275, 45, 44,       # 콧볼 확장
            239, 459, 309, 79       # 콧등 측면
        ],

        # [눈 확장] - 더 정밀한 보호 영역 (아이라인 포함)
        "EYE_L_EXTENDED": [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
            157, 158, 159, 160, 161, 246, 247, 30, 29, 27, 28, 56, 190,
            243, 112, 26, 22, 23, 24, 110, 25
        ],
        "EYE_R_EXTENDED": [
            263, 249, 390, 373, 374, 380, 381, 382, 362, 398,
            384, 385, 386, 387, 388, 466, 467, 260, 259, 257, 258, 286, 414,
            463, 341, 256, 252, 253, 254, 339, 255
        ],

        # [입술 확장] - 입술 주변 전체
        "LIPS_EXTENDED": [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
            # 입술 주변 피부
            57, 186, 92, 165, 167, 164, 393, 391, 322, 410,
            287, 273, 335, 406, 313, 18, 83, 182, 106, 43
        ],

        # [피부 영역] - 스무딩 적용 영역 (눈/입/눈썹 제외한 얼굴 내부)
        # 이것은 FACE_OVAL에서 보호 영역을 뺀 것을 런타임에 계산
        "SKIN_BOUNDARY": [
            # 이마 하단 (눈썹 위)
            108, 69, 104, 68, 71, 139, 34, 143, 156, 70,
            63, 105, 66, 107, 9, 336, 296, 334, 293, 300, 383, 264, 372, 385,
            # 볼 영역
            116, 117, 118, 119, 100, 36, 142, 126, 217, 174,
            345, 346, 347, 348, 329, 266, 371, 355, 437, 399,
            # 턱 라인 내부
            43, 106, 182, 83, 18, 313, 406, 335, 273
        ]
    }

    # [V25.0] 폴리곤 생성을 위한 순서 정렬된 인덱스
    POLYGON_INDICES = {
        # 눈 윤곽 (시계방향 정렬)
        "EYE_L_POLY": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144],
        "EYE_R_POLY": [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373],

        # 눈썹 윤곽 (정렬됨)
        "BROW_L_POLY": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        "BROW_R_POLY": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],

        # 입술 윤곽 (외곽 - 시계방향)
        "LIPS_OUTER_POLY": [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 409, 270, 269, 267, 0, 37, 39, 40, 185
        ],

        # 입술 윤곽 (내곽 - 시계방향)
        "LIPS_INNER_POLY": [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            308, 324, 318, 402, 317, 14, 87, 178, 88, 95
        ]
    }

    def __init__(self, root_dir=None):
        """
        [Mode A] MediaPipe Face Mesh Engine
        - 장점: 초경량(CPU 가능), 478개 상세 랜드마크
        - 변경: InsightFace(Heavy) -> MediaPipe(Light)
        """
        print("[AI] [FaceMesh] V25.0 High-Precision Polygon Masking Ready")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # 눈동자(Iris) 추적 포함
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[OK] [FaceMesh] 엔진 장전 완료 (CPU Optimized)")

    class FaceResult:
        """BeautyEngine과 호환성을 위한 결과 래퍼"""
        def __init__(self, landmarks):
            self.landmarks = landmarks # numpy array (478, 2)

    def process(self, frame_bgr):
        """
        :param frame_bgr: 입력 프레임 (BGR)
        :return: [FaceResult] 리스트
        """
        if frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]

        # 1. BGR -> RGB 변환 (MediaPipe는 RGB를 사용)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 2. 추론
        results = self.face_mesh.process(frame_rgb)

        final_faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 3. Normalized(0~1) -> Pixel(x,y) 변환
                # 속도를 위해 numpy 연산 사용
                lm_array = np.array([
                    [lm.x * w, lm.y * h] for lm in face_landmarks.landmark
                ], dtype=np.float32)  # V25.0: int -> float32 for precision

                final_faces.append(self.FaceResult(lm_array))

        return final_faces

    # ================================================================
    # [V25.0] 폴리곤 마스크 생성 메서드
    # ================================================================

    def get_face_polygon(self, landmarks, key="FACE_OVAL"):
        """
        지정된 영역의 폴리곤 좌표를 반환

        :param landmarks: (478, 2) numpy array
        :param key: FACE_INDICES 또는 POLYGON_INDICES의 키
        :return: (N, 2) numpy array of polygon vertices
        """
        if key in self.POLYGON_INDICES:
            indices = self.POLYGON_INDICES[key]
        elif key in self.FACE_INDICES:
            indices = self.FACE_INDICES[key]
        else:
            return None

        return landmarks[indices].astype(np.float32)

    def get_face_oval_polygon(self, landmarks):
        """얼굴 전체 윤곽 폴리곤 반환"""
        return self.get_face_polygon(landmarks, "FACE_OVAL")

    def get_exclusion_polygons(self, landmarks):
        """
        보호 영역 폴리곤들을 반환 (눈, 눈썹, 입술)

        :return: dict of polygon arrays
        """
        return {
            'eye_l': self.get_face_polygon(landmarks, "EYE_L_POLY"),
            'eye_r': self.get_face_polygon(landmarks, "EYE_R_POLY"),
            'brow_l': self.get_face_polygon(landmarks, "BROW_L_POLY"),
            'brow_r': self.get_face_polygon(landmarks, "BROW_R_POLY"),
            'lips': self.get_face_polygon(landmarks, "LIPS_OUTER_POLY")
        }

    def get_skin_mask_data(self, landmarks):
        """
        피부 마스크 생성에 필요한 모든 데이터를 반환

        :return: dict with face_oval and exclusion polygons
        """
        return {
            'face_oval': self.get_face_oval_polygon(landmarks),
            'exclusions': self.get_exclusion_polygons(landmarks)
        }

    def calculate_face_bounds(self, landmarks):
        """
        얼굴 영역의 바운딩 박스 계산

        :return: (x_min, y_min, x_max, y_max, center_x, center_y, radius)
        """
        face_pts = landmarks[self.FACE_INDICES["FACE_OVAL"]]

        x_min, y_min = np.min(face_pts, axis=0)
        x_max, y_max = np.max(face_pts, axis=0)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        radius = max(x_max - x_min, y_max - y_min) / 2

        return {
            'x_min': x_min, 'y_min': y_min,
            'x_max': x_max, 'y_max': y_max,
            'center_x': center_x, 'center_y': center_y,
            'radius': radius
        }

    def get_skin_sample_region(self, landmarks):
        """
        피부색 샘플링을 위한 안전한 영역 (볼 중앙) 반환

        :return: (x, y, width, height) of sampling region
        """
        # 양쪽 볼의 중심점 사용
        cheek_l_pts = landmarks[self.FACE_INDICES["CHEEK_L"]]
        cheek_r_pts = landmarks[self.FACE_INDICES["CHEEK_R"]]

        # 좌측 볼 중심
        cheek_l_center = np.mean(cheek_l_pts, axis=0)
        # 우측 볼 중심
        cheek_r_center = np.mean(cheek_r_pts, axis=0)

        # 전체 중심
        center = (cheek_l_center + cheek_r_center) / 2

        # 샘플링 영역 크기 (볼 영역의 약 20%)
        cheek_width = np.linalg.norm(cheek_l_center - cheek_r_center)
        sample_size = int(cheek_width * 0.15)

        return {
            'x': int(center[0] - sample_size / 2),
            'y': int(center[1] - sample_size / 2),
            'width': sample_size,
            'height': sample_size
        }

    # ================================================================
    # [기존 디버그 메서드 - 유지]
    # ================================================================

    def draw_debug(self, frame, faces):
        """
        [Visual Check] 랜드마크 시각화
        """
        if not faces:
            return frame

        for face in faces:
            # 주요 부위만 점 찍기
            for idx in range(0, len(face.landmarks), 2): # 너무 많으니 2개당 1개만
                pt = face.landmarks[idx].astype(int)
                cv2.circle(frame, tuple(pt), 1, (100, 255, 100), -1)

            # 코 끝 강조
            nose = face.landmarks[1].astype(int)
            cv2.circle(frame, tuple(nose), 4, (0, 0, 255), -1)

        return frame

    def draw_mesh_debug(self, frame, faces):
        return self.draw_debug(frame, faces)

    def draw_indices_debug(self, frame, faces):
        return self.draw_debug(frame, faces)

    # ================================================================
    # [V25.0] 폴리곤 디버그 시각화
    # ================================================================

    def draw_polygon_debug(self, frame, faces, show_exclusions=True):
        """
        폴리곤 마스크 영역 시각화
        """
        if not faces:
            return frame

        for face in faces:
            lm = face.landmarks

            # 얼굴 윤곽 (녹색)
            face_oval = self.get_face_oval_polygon(lm).astype(np.int32)
            cv2.polylines(frame, [face_oval], True, (0, 255, 0), 2)

            if show_exclusions:
                exclusions = self.get_exclusion_polygons(lm)

                # 눈 (빨간색)
                for key in ['eye_l', 'eye_r']:
                    pts = exclusions[key].astype(np.int32)
                    cv2.polylines(frame, [pts], True, (0, 0, 255), 1)

                # 눈썹 (노란색)
                for key in ['brow_l', 'brow_r']:
                    pts = exclusions[key].astype(np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 255), 1)

                # 입술 (분홍색)
                lips = exclusions['lips'].astype(np.int32)
                cv2.polylines(frame, [lips], True, (255, 0, 255), 1)

        return frame
