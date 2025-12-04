# Project MUSE - test_sam2.py (Interactive Mode for SAM 2)
# (C) 2025 MUSE Corp. All rights reserved.
# Purpose: SAM 2 (Segment Anything 2) Point-Prompt Demo
# "SAM 2의 성능을 테스트하기 위한 인터랙티브 툴 (Debug Mode + Smart Config Search)"

import os
import sys
import cv2
import numpy as np
import torch
import time
import traceback
import yaml # YAML 직접 파싱용 (Fallback)

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🔍 [System] 라이브러리 임포트 시작...")
try:
    import hydra
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import sam2
    print(f"   ✅ SAM 2 라이브러리 로드 성공 (Path: {os.path.dirname(sam2.__file__)})")
except ImportError as e:
    print(f"❌ [Critical] 라이브러리 로드 실패: {e}")
    print("   👉 'pip install git+https://github.com/facebookresearch/segment-anything-2.git'")
    traceback.print_exc()
    sys.exit(1)

# 전역 변수 (마우스 이벤트용)
click_point = None
clicked = False

def mouse_callback(event, x, y, flags, param):
    global click_point, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = np.array([x, y])
        clicked = True
        print(f"🖱️ [Click] 좌표 수신: ({x}, {y})")

def apply_background_removal(image, mask):
    if mask is None: return image
    
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
        
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    foreground = cv2.bitwise_and(image, mask_3ch)
    
    background_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background_dark = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
    background_dark = (background_dark * 0.3).astype(np.uint8)
    
    mask_inv = cv2.bitwise_not(mask)
    background_final = cv2.bitwise_and(background_dark, background_dark, mask=mask_inv)
    
    result = cv2.add(foreground, background_final)
    return result

def find_sam2_config_dir(target_config_name):
    """
    SAM 2 패키지 내부를 검색하여 설정 파일이 있는 '실제 폴더 경로'를 찾습니다.
    사용자 제보: site-packages/sam2/configs/sam2/sam2_hiera_l.yaml 처럼 중첩된 경우가 있음.
    """
    sam2_root = os.path.dirname(sam2.__file__)
    print(f"🔍 [Config Search] '{target_config_name}' 파일 탐색 중... (Root: {sam2_root})")
    
    found_dirs = []
    
    # os.walk로 모든 하위 폴더 검색
    for root, dirs, files in os.walk(sam2_root):
        if target_config_name in files:
            print(f"   -> 발견됨: {root}")
            found_dirs.append(root)
            
    if not found_dirs:
        return None
        
    # 우선순위: 경로에 'configs'가 포함된 곳을 선호 (구조적 정확성)
    # 예: sam2/configs/sam2/ > sam2/
    best_dir = found_dirs[0]
    for d in found_dirs:
        if "configs" in d:
            best_dir = d
            break # configs가 들어간 첫 번째 경로 선택
            
    print(f"   🎯 최종 선택된 설정 폴더: {best_dir}")
    return best_dir

def main():
    print("========================================================")
    print("   MUSE Interactive Tester (SAM 2 - Hiera Large)")
    print("   [DEBUG MODE ENABLED]")
    print("========================================================")

    # 1. 모델 경로 설정
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHECKPOINT = os.path.join(BASE_DIR, "assets", "models", "segment_anything", "sam2_hiera_large.pt")
    MODEL_CFG = "sam2_hiera_l.yaml"

    print(f"📂 모델 경로 확인: {CHECKPOINT}")
    if not os.path.exists(CHECKPOINT):
        print(f"❌ 파일이 존재하지 않습니다!")
        print("   👉 'tools/download_models.py' 실행 필요.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 실행 디바이스: {device}")

    # [Hydra Reset]
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
        print("🔧 [Config] 기존 Hydra 인스턴스 초기화(Clear) 완료")

    # 2. Config 경로 탐색 (Smart Search)
    config_dir = find_sam2_config_dir(MODEL_CFG)
    
    if not config_dir:
        print("❌ Config 파일을 찾을 수 없습니다.")
        print(f"   SAM 2 설치 경로({os.path.dirname(sam2.__file__)}) 안에 '{MODEL_CFG}' 파일이 있는지 확인하세요.")
        return

    # 3. SAM 2 빌드 (Absolute Path Strategy)
    print("\n⏳ [Build] SAM 2 모델 빌드 시작 (Absolute Config Path)...")
    sam2_model = None
    
    try:
        # [Strategy] initialize_config_dir 사용 (모듈 이름 대신 절대 경로 사용)
        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            print(f"   -> Hydra Config Dir: {config_dir}")
            
            try:
                # 1차 시도: 확장자 포함
                sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
            except Exception as e:
                print(f"   ⚠️ 1차 시도 실패: {e}")
                print("   🔄 2차 시도 (확장자 제거) 진행...")
                
                # 2차 시도: 확장자 제거
                cfg_name = MODEL_CFG.replace(".yaml", "")
                sam2_model = build_sam2(cfg_name, CHECKPOINT, device=device)

        print("   ✅ build_sam2 성공")
        
        predictor = SAM2ImagePredictor(sam2_model)
        print("   ✅ ImagePredictor 초기화 성공")
        
    except Exception as e:
        print("\n❌ [Fatal Error] 모델 빌드 실패")
        print("---------------- [ Traceback ] ----------------")
        traceback.print_exc()
        print("-----------------------------------------------")
        return

    # 4. 카메라 설정
    print("\n📷 [Camera] 카메라 연결 시도...")
    cap = cv2.VideoCapture(0)
    # MJPG 강제 설정 (USB 대역폭 확보)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ 카메라 연결 실패.")
        return
    print("   ✅ 카메라 연결됨.")

    window_name = "MUSE - SAM 2 Tester"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    global clicked

    while True:
        # [Live Mode]
        print("\n🎥 [Ready] 스페이스바를 눌러 캡처하세요...")
        captured_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            display = frame.copy()
            cv2.putText(display, "SAM 2 Ready - Press SPACE", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                captured_frame = frame
                break
        
        # [Encoding]
        print("📸 캡처됨! 인코딩 중...")
        frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
        
        try:
            t0 = time.time()
            predictor.set_image(frame_rgb)
            t1 = time.time()
            print(f"   ✅ 인코딩 완료 ({ (t1-t0)*1000:.1f}ms). 화면을 클릭하세요.")
        except Exception as e:
            print(f"❌ 인코딩 중 에러: {e}")
            traceback.print_exc()
            continue

        # [Interaction]
        clicked = False
        while True:
            display = captured_frame.copy()
            cv2.putText(display, "Click Object / R: Retry", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if clicked:
                try:
                    input_point = np.array([click_point])
                    input_label = np.array([1])
                    
                    print(f"✨ 추론 요청: {input_point}")
                    t0 = time.time()
                    masks, scores, _ = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True
                    )
                    t1 = time.time()
                    
                    best_idx = np.argmax(scores)
                    best_mask = masks[best_idx]
                    print(f"   -> 완료 ({ (t1-t0)*1000:.1f}ms). Score: {scores[best_idx]:.2f}")
                    
                    result_image = apply_background_removal(captured_frame, best_mask)
                    cv2.imshow("SAM 2 Result", result_image)
                    
                except Exception as e:
                    print(f"❌ 추론 실패: {e}")
                    traceback.print_exc()
                
                clicked = False
            
            cv2.imshow(window_name, display)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('r'):
                cv2.destroyWindow("SAM 2 Result")
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()