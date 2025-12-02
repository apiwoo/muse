# Project MUSE - download_models.py
# (C) 2025 MUSE Corp. All rights reserved.
# Target: RTX 3060+ (High-End Models) & FFmpeg NVDEC Setup

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
LIBS_DIR = os.path.join(BASE_DIR, "libs") # FFmpeg 설치 경로

# 1. ViTPose-Huge (Body Tracking) - 가장 무겁고 정확한 모델
# [수정] HuggingFace Public Mirror 사용 (JunkyByte/easy_ViTPose)
VITPOSE_URL = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/coco/vitpose-h-coco.pth"
VITPOSE_DIR = os.path.join(MODEL_ROOT, "tracking")
VITPOSE_PATH = os.path.join(VITPOSE_DIR, "vitpose_huge_coco_256x192.pth")

# 2. InsightFace Buffalo_L (Face Analysis) - 고정밀 얼굴 분석 팩
INSIGHTFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
INSIGHTFACE_DIR = os.path.join(MODEL_ROOT, "insightface")
INSIGHTFACE_ZIP = os.path.join(INSIGHTFACE_DIR, "buffalo_l.zip")

# 3. [New] FFmpeg (NVDEC Build) - BtbN Auto-Build (Windows x64)
# GPL Shared/Static 상관없이 exe만 있으면 되므로, 가장 널리 쓰이는 BtbN 빌드 사용
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
FFMPEG_ZIP = os.path.join(LIBS_DIR, "ffmpeg_temp.zip")
FFMPEG_EXE_TARGET = os.path.join(LIBS_DIR, "ffmpeg.exe")

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
        block_size = 1024 * 1024 # 1MB 청크
        
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
        return True
    except Exception as e:
        print(f"   ❌ 압축 해제 실패: {e}")
        return False

def setup_ffmpeg():
    """
    [New] FFmpeg ZIP에서 ffmpeg.exe만 추출하여 libs 폴더에 배치
    """
    print("🚀 [Step 3] FFmpeg (NVDEC GPU Accelerated) 준비...")
    
    # 이미 설치되어 있는지 확인
    if os.path.exists(FFMPEG_EXE_TARGET):
        print(f"   ✅ 이미 설치됨: {FFMPEG_EXE_TARGET}")
        return

    # libs 폴더 생성
    os.makedirs(LIBS_DIR, exist_ok=True)

    # 다운로드
    if not download_file(FFMPEG_URL, FFMPEG_ZIP):
        print("   ❌ FFmpeg 다운로드 실패. 인터넷 연결을 확인하세요.")
        return

    # 압축 해제 및 exe 추출 (cherry-pick)
    print("   📦 FFmpeg 추출 중 (ffmpeg.exe만 꺼냅니다)...")
    try:
        with zipfile.ZipFile(FFMPEG_ZIP, 'r') as zf:
            # zip 내부 구조: ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe
            # 파일을 찾아서 추출
            found = False
            for file_info in zf.infolist():
                if file_info.filename.endswith("bin/ffmpeg.exe"):
                    # 임시 경로에 추출
                    file_info.filename = "ffmpeg.exe" # 이름 변경
                    zf.extract(file_info, LIBS_DIR)
                    found = True
                    break
            
            if found:
                print(f"   ✅ FFmpeg 설치 완료: {FFMPEG_EXE_TARGET}")
            else:
                print("   ❌ ZIP 파일 내에서 ffmpeg.exe를 찾을 수 없습니다.")
    
    except Exception as e:
        print(f"   ❌ FFmpeg 설치 중 오류: {e}")
    
    # 임시 ZIP 파일 삭제
    if os.path.exists(FFMPEG_ZIP):
        os.remove(FFMPEG_ZIP)

# =========================================================
# [메인] 실행 로직
# =========================================================

def main():
    print("============================================================")
    print("   Project MUSE - High-End Model & Tool Downloader")
    print("============================================================")
    print(f"📂 모델 저장 경로: {MODEL_ROOT}")
    print(f"📂 도구 저장 경로: {LIBS_DIR}\n")

    # 1. ViTPose-Huge 다운로드
    print("🚀 [Step 1] ViTPose-Huge (Body Tracking Model) 준비...")
    if download_file(VITPOSE_URL, VITPOSE_PATH):
        print("   -> Body Engine 준비 완료.\n")
    else:
        print("   -> Body Engine 준비 실패.\n")

    # 2. InsightFace Buffalo_L 다운로드
    print("🚀 [Step 2] InsightFace Buffalo_L (Face Analysis Model) 준비...")
    os.makedirs(INSIGHTFACE_DIR, exist_ok=True)
    check_file = os.path.join(INSIGHTFACE_DIR, "1k3d68.onnx")
    if os.path.exists(check_file):
        print("   ✅ 이미 설치됨: InsightFace Models\n")
    else:
        if download_file(INSIGHTFACE_URL, INSIGHTFACE_ZIP):
            extract_zip(INSIGHTFACE_ZIP, INSIGHTFACE_DIR)
            print("   -> Face Engine 준비 완료.\n")
        else:
            print("   -> Face Engine 준비 실패.\n")

    # 3. [New] FFmpeg 다운로드
    setup_ffmpeg()

    print("\n============================================================")
    print("🎉 모든 필수 파일(AI 모델 + FFmpeg) 준비 완료.")
    print("👉 이제 GPU 가속을 위한 'src/core/input_manager.py' 수정이 가능합니다.")
    print("============================================================")

if __name__ == "__main__":
    main()