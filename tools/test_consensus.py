# Project MUSE - test_consensus.py
# Tri-Core Fusion Engine Tester (Diagnosis Mode)
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import time
import cv2
import numpy as np
import traceback

# -----------------------------------------------------------
# [System Setup] 프로젝트 경로 및 CUDA 라이브러리 설정
# -----------------------------------------------------------
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)

if os.path.basename(current_dir) == 'tools':
    sys.path.append(parent_dir)
else:
    sys.path.append(current_dir)

try:
    import cupy as cp
    HAS_CUDA = True
    print(f"🚀 [System] CuPy Loaded: Device {cp.cuda.runtime.getDevice()}")
except ImportError:
    HAS_CUDA = False
    print("❌ [Critical] CuPy not found. This engine requires NVIDIA GPU.")
    sys.exit(1)

from src.utils.cuda_helper import setup_cuda_environment
setup_cuda_environment()

from src.ai.consensus_engine import ConsensusEngine

# -----------------------------------------------------------
# [Visualization] 뼈대 그리기 유틸리티
# -----------------------------------------------------------
SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]

def draw_skeleton(img, kpts, conf_thresh=0.3):
    if kpts is None: return img
    vis_img = img.copy()
    for p1, p2 in SKELETON_EDGES:
        if p1 < len(kpts) and p2 < len(kpts):
            x1, y1, c1 = kpts[p1]
            x2, y2, c2 = kpts[p2]
            if c1 > conf_thresh and c2 > conf_thresh:
                cv2.line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    for i, (x, y, c) in enumerate(kpts):
        if c > conf_thresh:
            color = (0, 0, 255) if i < 5 else (0, 255, 0)
            cv2.circle(vis_img, (int(x), int(y)), 4, color, -1)
    return vis_img

def main():
    print("========================================================")
    print("   MUSE Tri-Core Engine Tester (Diagnosis Mode)")
    print("========================================================")

    if os.path.basename(current_dir) == 'tools':
        root_dir = parent_dir
    else:
        root_dir = current_dir
    
    # 1. 엔진 초기화
    print("\n⏳ [Init] Consensus Engine 로딩 중...")
    try:
        engine = ConsensusEngine(root_dir)
        print("   ✅ 엔진 초기화 완료.")
    except Exception as e:
        print(f"❌ 엔진 로드 실패: {e}")
        traceback.print_exc()
        return

    # 2. 카메라 설정 (720p + DSHOW)
    print("\n📷 [Camera] 카메라 연결 시도... (Backend: DirectShow)")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ 카메라 연결 실패.")
        return
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   ✅ 카메라 연결됨. Resolution: {w}x{h}")
    
    # 3. 배경 이미지 (Green Screen)
    TARGET_W, TARGET_H = 1920, 1080
    bg_green = np.full((TARGET_H, TARGET_W, 3), (0, 255, 0), dtype=np.uint8)

    print("\n🚀 [Start] 진단 모드 시작.")
    print("   [Controls]")
    print("   'Q': 종료")
    print("   'C': RGB/BGR 모드 전환 (현재: BGR Default)")
    
    prev_time = time.time()
    frame_cnt = 0
    fps = 0
    use_rgb = False # 색상 모드 토글

    while True:
        ret, frame_orig = cap.read()
        if not ret:
            # 프레임 수신 실패 시 빨간 화면 표시 (진단용)
            error_screen = np.full((540, 960, 3), (0, 0, 255), dtype=np.uint8)
            cv2.putText(error_screen, "CAMERA ERROR / NO SIGNAL", (50, 270), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("MUSE Consensus Tester", error_screen)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        t0 = time.perf_counter()

        # [Pipeline Step 0] Resize
        frame = cv2.resize(frame_orig, (TARGET_W, TARGET_H))
        
        # [Debug] Color Space Toggle
        if use_rgb:
            frame_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_input = frame

        # [Pipeline Step 1] Upload
        frame_gpu = cp.asarray(frame_input)

        # [Pipeline Step 2] AI Processing
        alpha_matte_gpu, keypoints = engine.process(frame_gpu)
        
        t1 = time.perf_counter()
        inference_time = (t1 - t0) * 1000

        # [Pipeline Step 3] Visualization Logic
        alpha_max = 0.0
        
        if alpha_matte_gpu is not None:
            alpha_cpu = alpha_matte_gpu.get()
            alpha_max = float(alpha_cpu.max()) # 진단용 Max 값
            
            alpha_3c = np.repeat(alpha_cpu[:, :, np.newaxis], 3, axis=2)
            
            # 합성
            foreground = frame.astype(np.float32) * alpha_3c
            background = bg_green.astype(np.float32) * (1.0 - alpha_3c)
            combined = (foreground + background).astype(np.uint8)
            
            # 마스크 뷰
            mask_view = (alpha_cpu * 255).astype(np.uint8)
            mask_view = cv2.cvtColor(mask_view, cv2.COLOR_GRAY2BGR)
        else:
            combined = frame.copy()
            mask_view = np.zeros_like(frame)

        # [Diagnosis View Construction] 4분할 화면 생성
        # 1. 원본 (Skeleton Overlay)
        raw_view = frame.copy()
        if keypoints is not None:
            raw_view = draw_skeleton(raw_view, keypoints)
            
        # 2. 마스크 (원본 크기)
        
        # 3. 합성 (원본 크기)

        # 리사이징 (쿼드 뷰용)
        disp_w, disp_h = 640, 360
        
        view_tl = cv2.resize(combined, (disp_w, disp_h)) # Top-Left: Result
        view_tr = cv2.resize(mask_view, (disp_w, disp_h)) # Top-Right: Mask
        view_bl = cv2.resize(frame, (disp_w, disp_h))    # Bot-Left: Raw Input
        view_br = cv2.resize(raw_view, (disp_w, disp_h)) # Bot-Right: Skeleton Logic
        
        # 라벨링
        cv2.putText(view_tl, f"1. Result (FPS: {fps})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(view_tr, f"2. Alpha Mask (Max: {alpha_max:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(view_bl, "3. Raw Input (Check Camera)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(view_br, "4. ViTPose Debug", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        # 그리드 병합
        top_row = np.hstack([view_tl, view_tr])
        bot_row = np.hstack([view_bl, view_br])
        final_display = np.vstack([top_row, bot_row])

        # Color Mode 표시
        mode_text = "RGB Mode" if use_rgb else "BGR Mode (Default)"
        cv2.putText(final_display, f"Mode: {mode_text} (Press 'C' to switch)", (20, final_display.shape[0]-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("MUSE Consensus Tester", final_display)

        # FPS & Log
        frame_cnt += 1
        curr_time = time.time()
        if curr_time - prev_time >= 1.0:
            fps = frame_cnt
            frame_cnt = 0
            prev_time = curr_time
            print(f"[Log] FPS: {fps} | Alpha Max: {alpha_max:.4f} | Kpts: {'YES' if keypoints is not None else 'NO'}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            use_rgb = not use_rgb
            print(f"[Input] Switched to {'RGB' if use_rgb else 'BGR'} mode")

    cap.release()
    cv2.destroyAllWindows()
    print("[Exit] 테스트 종료.")

if __name__ == "__main__":
    main()