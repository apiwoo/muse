# Project MUSE - download_models.py
# (C) 2025 MUSE Corp. All rights reserved.
# Target: RTX 3060+ (High-End Models)

import os
import requests
import zipfile
import shutil
from tqdm import tqdm

# =========================================================
# [설정] 다운로드 경로 및 모델 URL
# =========================================================

# 프로젝트 루트 기준 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.path.join(BASE_DIR, "assets", "models")

# 1. ViTPose-Huge (Body Tracking) - 가장 무겁고 정확한 모델
# [수정] HuggingFace Public Mirror 사용 (JunkyByte/easy_ViTPose)
# 원본 msp/ViTPose는 인증 필요, 이전 경로는 404 에러 발생.
# 확인된 최신 경로(torch/coco/vitpose-h-coco.pth)로 변경합니다.
# (vitpose-h는 Huge 모델을 의미합니다. 약 2.5GB)
VITPOSE_URL = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/coco/vitpose-h-coco.pth"
VITPOSE_DIR = os.path.join(MODEL_ROOT, "tracking")
# 저장 파일명은 프로젝트 호환성을 위해 'vitpose_huge_coco_256x192.pth'로 유지합니다.
VITPOSE_PATH = os.path.join(VITPOSE_DIR, "vitpose_huge_coco_256x192.pth")

# 2. InsightFace Buffalo_L (Face Analysis) - 고정밀 얼굴 분석 팩
# Github Release 사용
INSIGHTFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
INSIGHTFACE_DIR = os.path.join(MODEL_ROOT, "insightface")
INSIGHTFACE_ZIP = os.path.join(INSIGHTFACE_DIR, "buffalo_l.zip")

# =========================================================
# [유틸리티] 다운로드 함수
# =========================================================

def download_file(url, dest_path):
    """
    URL에서 파일을 다운로드하며 진행률(ProgressBar)을 표시합니다.
    """
    if os.path.exists(dest_path):
        print(f"   ✅ 이미 존재함: {os.path.basename(dest_path)}")
        return True

    print(f"   ⬇️ 다운로드 시작: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1KB
        
        # 폴더가 없으면 생성
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        
        print(f"   ✅ 다운로드 완료: {dest_path}")
        return True
    
    except Exception as e:
        print(f"   ❌ 다운로드 실패: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path) # 실패한 파일 삭제
        return False

def extract_zip(zip_path, extract_to):
    """
    ZIP 파일을 압축 해제합니다.
    """
    print(f"   📦 압축 해제 중: {os.path.basename(zip_path)}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"   ✅ 압축 해제 완료: {extract_to}")
        
        # 압축 해제 후 ZIP 파일 삭제 (선택 사항)
        # os.remove(zip_path) 
        return True
    except Exception as e:
        print(f"   ❌ 압축 해제 실패: {e}")
        return False

# =========================================================
# [메인] 실행 로직
# =========================================================

def main():
    print("============================================================")
    print("   Project MUSE - High-End Model Downloader")
    print("============================================================")
    print(f"📂 모델 저장 경로: {MODEL_ROOT}\n")

    # 1. ViTPose-Huge 다운로드
    print("🚀 [Step 1] ViTPose-Huge (Body Tracking Model) 준비...")
    if download_file(VITPOSE_URL, VITPOSE_PATH):
        print("   -> Body Engine 준비 완료.\n")
    else:
        print("   -> Body Engine 준비 실패. 인터넷 연결을 확인하세요.\n")

    # 2. InsightFace Buffalo_L 다운로드
    print("🚀 [Step 2] InsightFace Buffalo_L (Face Analysis Model) 준비...")
    # 폴더 생성
    os.makedirs(INSIGHTFACE_DIR, exist_ok=True)
    
    # 이미 압축 해제된 파일이 있는지 확인 (대표 파일: 1k3d68.onnx)
    check_file = os.path.join(INSIGHTFACE_DIR, "1k3d68.onnx")
    if os.path.exists(check_file):
        print("   ✅ 이미 설치됨: InsightFace Models")
    else:
        if download_file(INSIGHTFACE_URL, INSIGHTFACE_ZIP):
            extract_zip(INSIGHTFACE_ZIP, INSIGHTFACE_DIR)
        else:
            print("   -> Face Engine 준비 실패.\n")

    print("============================================================")
    print("🎉 모든 모델 준비 완료.")
    print("👉 이제 'src/ai/tracking/' 코드를 구현할 차례입니다.")
    print("============================================================")

if __name__ == "__main__":
    main()