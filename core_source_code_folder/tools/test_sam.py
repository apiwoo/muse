# Project MUSE - test_sam.py (Interactive Mode)
# (C) 2025 MUSE Corp. All rights reserved.
# Purpose: SAM (Segment Anything Model) Point-Prompt Demo
# "클릭 한 번으로 배경 날리기 (Prompt-based Segmentation)"

import os
import sys
import cv2
import numpy as np
import torch
import time

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # [핵심 변경] AutomaticGenerator 대신 Predictor를 사용합니다.
    # Predictor는 사용자의 '힌트(점, 박스)'를 기다리는 녀석입니다.
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("❌ 'segment_anything' 라이브러리가 없습니다.")
    sys.exit(1)

# 전역 변수 (마우스 이벤트용)
click_point = None
clicked = False

def mouse_callback(event, x, y, flags, param):
    global click_point, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = np.array([x, y])
        clicked = True
        print(f"🖱️ 클릭 좌표 수신: ({x}, {y})")

def apply_background_removal(image, mask):
    """
    마스크 영역만 컬러로 남기고, 배경은 흑백(또는 검은색)으로 처리하여 강조
    """
    # 마스크를 3채널로 확장 (True/False -> 0/1 -> 0/255)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    
    # 1. 전경 (사람): 마스크가 True인 부분은 원본 이미지 사용
    foreground = np.where(mask_3ch, image, 0)
    
    # 2. 배경 (나머지): 검은색으로 날려버리기 (크로마키 효과)
    # background = np.zeros_like(image) 
    
    # (옵션) 배경을 아예 없애지 않고 흐릿하게 보고 싶다면 아래 주석 해제
    # background = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR) // 0.3
    
    # 합성
    result = foreground # + background.astype(np.uint8)
    return result

def main():
    print("========================================================")
    print("   MUSE Interactive Segmentation (SAM ViT-Huge)")
    print("========================================================")
    print("   1. 웹캠이 켜지면 'SPACE'를 눌러 화면을 캡처(Freeze)하세요.")
    print("   2. 멈춘 화면에서 본인(또는 원하는 물체)을 '클릭'하세요.")
    print("   3. SAM이 클릭된 물체만 인식해서 배경을 날려버립니다.")
    print("========================================================")

    # 1. 모델 설정
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "assets", "models", "segment_anything", "sam_vit_h_4b8939.pth")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")

    # 2. 모델 로딩
    print("⏳ SAM 모델(ViT-Huge) 로딩 중... (무거운 모델입니다)")
    sam = sam_model_registry["vit_h"](checkpoint=MODEL_PATH)
    sam.to(device=device)
    
    # [핵심] Predictor 초기화
    predictor = SamPredictor(sam)
    print("✅ 모델 준비 완료. (Interactive Mode)")

    # 3. 카메라 설정
    cap = cv2.VideoCapture(1)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 윈도우 및 마우스 콜백 설정
    window_name = "MUSE - Interactive SAM"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    global clicked
    
    while True:
        # ------------------------------------------------
        # [Mode 1] Live Camera Loop (촬영 대기)
        # ------------------------------------------------
        print("\n🎥 [Live Mode] SPACE를 눌러 캡처하세요.")
        captured_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # 가이드 텍스트
            display = frame.copy()
            cv2.putText(display, "Live View - Press SPACE to Capture", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '): # 스페이스바
                captured_frame = frame
                break
        
        # ------------------------------------------------
        # [Mode 2] Image Encoding (선생님의 생각 시간)
        # ------------------------------------------------
        print("📸 캡처됨! SAM이 이미지를 분석합니다... (Encoding)")
        
        # SAM은 RGB를 좋아합니다.
        frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
        
        t0 = time.time()
        # [핵심] 이미지 전체를 한 번 읽어서 '특징(Embedding)'을 추출합니다.
        # 이 과정은 무겁지만(약 0.5~1초), 한 번만 하면 됩니다.
        predictor.set_image(frame_rgb)
        t1 = time.time()
        print(f"✅ 인코딩 완료 (소요시간: {t1 - t0:.2f}초).")
        print("👉 이제 화면 속 원하는 물체를 '클릭'하세요!")

        # ------------------------------------------------
        # [Mode 3] Interaction Loop (클릭 대기)
        # ------------------------------------------------
        clicked = False # 상태 초기화
        
        while True:
            display = captured_frame.copy()
            cv2.putText(display, "Click Object / Press R to Retry", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # 마우스 클릭이 감지되면 실행
            if clicked:
                # 1. 입력 좌표 준비
                input_point = np.array([click_point])
                input_label = np.array([1]) # 1은 '이거야(Foreground)', 0은 '이거 아냐(Background)'
                
                print(f"✨ [Prompt] 좌표 {input_point} 추론 요청...")
                
                # 2. 마스크 예측 (Decoder 실행 - 매우 빠름)
                # 인코딩된 정보를 바탕으로 마스크만 뱉어냅니다.
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True, # 애매할 경우 3가지 후보를 줍니다.
                )
                
                # 3. 가장 점수(IoU)가 높은 마스크 선택
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]
                
                print(f"   -> 완료! (Score: {scores[best_idx]:.2f})")
                
                # 4. 결과 시각화 (배경 날리기)
                result_image = apply_background_removal(captured_frame, best_mask)
                
                # 결과창 띄우기
                cv2.imshow("MUSE - Result", result_image)
                print("   -> 'MUSE - Result' 창을 확인하세요.")
                print("   -> (아무 키나 누르면 결과창이 닫힙니다)")
                cv2.waitKey(0)
                cv2.destroyWindow("MUSE - Result")
                
                clicked = False # 다시 클릭 대기
                print("   -> 다른 물체를 클릭하거나, 'R'을 눌러 다시 찍으세요.")

            cv2.imshow(window_name, display)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('r'): # 재촬영
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()