# Project MUSE - download_models.py
# (C) 2025 MUSE Corp. All rights reserved.
# Target: SAM 2, ViTPose (Huge Only), MODNet (CKPT & ONNX)
# Updated: Added robust Google Drive support without gdown dependency

import os
import requests
import zipfile
import shutil
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.path.join(BASE_DIR, "assets", "models")
LIBS_DIR = os.path.join(BASE_DIR, "libs")

# 1. ViTPose Model (Huge Only)
# [Runtime] Huge: 정확도 최우선 (실시간 + 라벨링용)
VITPOSE_HUGE_URL = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/coco/vitpose-h-coco.pth"
VITPOSE_HUGE_PATH = os.path.join(MODEL_ROOT, "tracking", "vitpose_huge_coco_256x192.pth")

# 2. SAM 2 Hiera-Large (Teacher for Segmentation - Video)
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
SAM2_PATH = os.path.join(MODEL_ROOT, "segment_anything", "sam2_hiera_large.pt")

# 3. SAM 2.1 Models
SAM2_TINY_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
SAM2_TINY_PATH = os.path.join(MODEL_ROOT, "segment_anything", "sam2.1_hiera_tiny.pt")
SAM2_1_LARGE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_1_LARGE_PATH = os.path.join(MODEL_ROOT, "segment_anything", "sam2.1_hiera_large.pt")

# 4. MODNet (Segmentation)
# [Training Base] PyTorch Checkpoint (CKPT) - 필수
# Google Drive ID: 1mcr7ALciuAsHCpLnrtG_eop5jLhb0et_
MODNET_CKPT_MIRRORS = [
    # Mirror 1: Google Drive (Official) - Handled by specific logic
    "gdrive:1mcr7ALciuAsHCpLnrtG_eop5jLhb0et_",
    # Mirror 2: Hugging Face Spaces (Public Access)
    "https://huggingface.co/spaces/nielsr/MODNet/resolve/main/modnet_webcam_portrait_matting.ckpt",
    # Mirror 3: Backup GitHub Release
    "https://github.com/ZHKKKe/MODNet/releases/download/v1.0.0/modnet_webcam_portrait_matting.ckpt"
]
MODNET_CKPT_PATH = os.path.join(MODEL_ROOT, "segmentation", "modnet_webcam_portrait_matting.ckpt")

# [Runtime Base] ONNX (Webcam Optimized)
MODNET_ONNX_URL = "https://github.com/Zeyi-Lin/HivisionIDPhotos/releases/download/pretrained-model/modnet_photographic_portrait_matting.onnx"
MODNET_ONNX_PATH = os.path.join(MODEL_ROOT, "segmentation", "modnet.onnx")

# 5. InsightFace & FFmpeg
INSIGHTFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
INSIGHTFACE_DIR = os.path.join(MODEL_ROOT, "insightface")
INSIGHTFACE_ZIP = os.path.join(INSIGHTFACE_DIR, "buffalo_l.zip")

FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
FFMPEG_ZIP = os.path.join(LIBS_DIR, "ffmpeg_temp.zip")
FFMPEG_EXE_TARGET = os.path.join(LIBS_DIR, "ffmpeg.exe")

def download_file(url, dest_path):
    """Single file download"""
    if os.path.exists(dest_path):
        if os.path.getsize(dest_path) < 1024:
            print(f"   [WARN] Corrupted file detected: {os.path.basename(dest_path)}. Redownloading...")
            os.remove(dest_path)
        else:
            print(f"   [OK] Already exists: {os.path.basename(dest_path)}")
            return True

    print(f"   [DOWN] Downloading: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 

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
        
        print(f"   [OK] Download Complete: {dest_path}")
        return True
    except Exception as e:
        print(f"   [ERROR] Download Failed: {e}")
        if os.path.exists(dest_path): os.remove(dest_path)
        return False

def download_from_google_drive(id, dest_path):
    """Download file from Google Drive using requests (gdown alternative)"""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    try:
        response = session.get(URL, params={'id': id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
        
        # Check if valid (not HTML error page)
        if os.path.getsize(dest_path) < 2000:
            # Maybe it failed and downloaded a small HTML file
            with open(dest_path, 'r', errors='ignore') as f:
                head = f.read(100)
                if "<html" in head.lower():
                    print("   [ERROR] Google Drive download failed (Quota exceeded or Auth required).")
                    return False
        
        print(f"   [OK] GDrive Download Complete: {dest_path}")
        return True
    except Exception as e:
        print(f"   [ERROR] GDrive Download Failed: {e}")
        return False

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def download_with_fallback(urls, dest_path):
    """Try multiple URLs until one succeeds"""
    if os.path.exists(dest_path):
        if os.path.getsize(dest_path) > 1024:
            print(f"   [OK] Already exists: {os.path.basename(dest_path)}")
            return True
        else:
            os.remove(dest_path)

    print(f"   [MULTI] Attempting download for: {os.path.basename(dest_path)}")
    for i, url in enumerate(urls):
        print(f"      -> Attempt {i+1}/{len(urls)}: {url}")
        
        if url.startswith("gdrive:"):
            g_id = url.split(":")[1]
            if download_from_google_drive(g_id, dest_path):
                return True
        else:
            if download_file(url, dest_path):
                return True
        
        print("      -> Failed. Trying next mirror...")
    
    print(f"   [FAIL] All mirrors failed for {os.path.basename(dest_path)}")
    return False

def extract_zip(zip_path, extract_to):
    print(f"   [PKG] Extracting: {os.path.basename(zip_path)}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"   [OK] Extraction Complete")
        return True
    except Exception as e:
        print(f"   [ERROR] Extraction Failed: {e}")
        return False

def setup_ffmpeg():
    if os.path.exists(FFMPEG_EXE_TARGET): return
    os.makedirs(LIBS_DIR, exist_ok=True)
    if download_file(FFMPEG_URL, FFMPEG_ZIP):
        print("   [PKG] Extracting FFmpeg...")
        try:
            with zipfile.ZipFile(FFMPEG_ZIP, 'r') as zf:
                for file_info in zf.infolist():
                    if file_info.filename.endswith("bin/ffmpeg.exe"):
                        file_info.filename = "ffmpeg.exe"
                        zf.extract(file_info, LIBS_DIR)
                        print(f"   [OK] FFmpeg Installed: {FFMPEG_EXE_TARGET}")
                        break
        except Exception as e:
            print(f"   [ERROR] FFmpeg Install Error: {e}")
        if os.path.exists(FFMPEG_ZIP): os.remove(FFMPEG_ZIP)

def main():
    print("============================================================")
    print("   MUSE Model Downloader (Robust Mirror Mode)")
    print("============================================================")
    
    # Ensure directories
    os.makedirs(os.path.dirname(SAM2_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(VITPOSE_HUGE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODNET_CKPT_PATH), exist_ok=True)

    # 1. ViTPose (Pose Model - Huge Only)
    print("\n>>> Checking Pose Model (Huge)...")
    download_file(VITPOSE_HUGE_URL, VITPOSE_HUGE_PATH)
    
    # 2. SAM 2 (Teacher Models)
    print("\n>>> Checking SAM 2 Models...")
    download_file(SAM2_URL, SAM2_PATH)
    download_file(SAM2_TINY_URL, SAM2_TINY_PATH)
    download_file(SAM2_1_LARGE_URL, SAM2_1_LARGE_PATH)
    
    # 3. MODNet (Segmentation Models)
    print("\n>>> Checking MODNet Models...")
    # [Fix] Use Fallback downloader for MODNet CKPT
    download_with_fallback(MODNET_CKPT_MIRRORS, MODNET_CKPT_PATH)
    download_file(MODNET_ONNX_URL, MODNET_ONNX_PATH)
    
    # 4. Utilities
    print("\n>>> Checking Utilities...")
    os.makedirs(INSIGHTFACE_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(INSIGHTFACE_DIR, "1k3d68.onnx")):
        if download_file(INSIGHTFACE_URL, INSIGHTFACE_ZIP):
            extract_zip(INSIGHTFACE_ZIP, INSIGHTFACE_DIR)
            
    setup_ffmpeg()
    
    print("\n[DONE] All models ready.")

if __name__ == "__main__":
    main()