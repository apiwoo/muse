# Project MUSE - download_models.py
# (C) 2025 MUSE Corp. All rights reserved.
# Target: SAM 2 (Hiera-Large) & ViTPose & SegFormer Weights

import os
import requests
import zipfile
import shutil
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.path.join(BASE_DIR, "assets", "models")
LIBS_DIR = os.path.join(BASE_DIR, "libs")

# 1. ViTPose-Huge (Teacher for Pose)
VITPOSE_URL = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/coco/vitpose-h-coco.pth"
VITPOSE_PATH = os.path.join(MODEL_ROOT, "tracking", "vitpose_huge_coco_256x192.pth")

# 2. SAM 2 Hiera-Large (Teacher for Segmentation - Video)
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
SAM2_PATH = os.path.join(MODEL_ROOT, "segment_anything", "sam2_hiera_large.pt")

# 3. SegFormer (MiT-B1)
SEGFORMER_DIR = os.path.join(MODEL_ROOT, "pretrained")

# 4. InsightFace
INSIGHTFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
INSIGHTFACE_DIR = os.path.join(MODEL_ROOT, "insightface")
INSIGHTFACE_ZIP = os.path.join(INSIGHTFACE_DIR, "buffalo_l.zip")

# 5. FFmpeg
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
FFMPEG_ZIP = os.path.join(LIBS_DIR, "ffmpeg_temp.zip")
FFMPEG_EXE_TARGET = os.path.join(LIBS_DIR, "ffmpeg.exe")

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"   [OK] Already exists: {os.path.basename(dest_path)}")
        return True

    print(f"   [DOWN] Downloading: {url}")
    try:
        response = requests.get(url, stream=True)
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
    print("   MUSE Model Downloader (v1.1 SAM2 Ready)")
    print("============================================================")
    
    os.makedirs(os.path.dirname(SAM2_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(VITPOSE_PATH), exist_ok=True)
    os.makedirs(SEGFORMER_DIR, exist_ok=True)

    download_file(VITPOSE_URL, VITPOSE_PATH)
    download_file(SAM2_URL, SAM2_PATH)
    
    os.makedirs(INSIGHTFACE_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(INSIGHTFACE_DIR, "1k3d68.onnx")):
        if download_file(INSIGHTFACE_URL, INSIGHTFACE_ZIP):
            extract_zip(INSIGHTFACE_ZIP, INSIGHTFACE_DIR)
            
    setup_ffmpeg()
    
    print("\n[DONE] All models ready.")

if __name__ == "__main__":
    main()