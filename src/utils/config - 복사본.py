# Project MUSE - src/utils/create_assets.py
# Created for AI Beauty Cam Project
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import numpy as np
import mediapipe as mp

# 프로젝트 루트 경로 설정 (상위 폴더 인식)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from src.utils.logger import get_logger

def create_triangulation_data():
    logger = get_logger("AssetCreator")
    logger.info("📐 렌더링용 삼각형 인덱스 데이터 생성을 시작합니다...")

    # 저장 경로 확인
    save_dir = os.path.join(project_root, "assets", "data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info(f"폴더 생성됨: {save_dir}")

    save_path = os.path.join(save_dir, "triangulation.npy")

    # 1. MediaPipe에서 삼각형 연결 정보(Tesselation) 가져오기
    mp_face_mesh = mp.solutions.face_mesh
    
    # frozenset을 리스트로 변환 후 정렬 (일관성 유지)
    # FACEMESH_TESSELATION은 (start_index, end_index) 선분들의 집합입니다.
    # 하지만 OpenGL 렌더링을 위해서는 '점 3개'로 이루어진 삼각형 인덱스가 필요합니다.
    # MediaPipe는 기본적으로 삼각형 리스트를 제공하지 않으므로, 
    # 여기서는 렌더링에 가장 적합한 'FACEMESH_TESSELATION' 기반으로 추후 가공하거나,
    # 우선은 단순히 점 연결 확인용으로 원본 데이터를 저장합니다.
    
    # [참고] 실제 3D 렌더링을 위해서는 정점 3개씩 묶인 인덱스 배열이 필요합니다.
    # 여기서는 MediaPipe의 표준 토폴로지를 저장합니다.
    tesselation = list(mp_face_mesh.FACEMESH_TESSELATION)
    
    # numpy 배열로 변환 (N, 2) - 선분 데이터
    # 나중에 renderer.py에서 이를 이용해 Wireframe을 그리거나, 
    # 별도의 알고리즘으로 삼각형을 구성하게 됩니다.
    data = np.array(tesselation, dtype=np.int32)
    
    # 2. 파일 저장
    np.save(save_path, data)
    
    logger.info(f"✅ 데이터 저장 완료: {save_path}")
    logger.info(f"👉 데이터 크기: {data.shape} (선분 개수: {len(data)})")
    logger.info("이 파일은 추후 'Graphics' 모듈에서 얼굴 표면을 그릴 때 사용됩니다.")

if __name__ == "__main__":
    create_triangulation_data()