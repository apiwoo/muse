# Project MUSE - test_camera_pure.py
# (C) 2025 MUSE Corp. All rights reserved.
# "GPU, AI 다 끄고 오직 카메라만 테스트합니다."

import cv2
import time
import sys

def main():
    print("========================================")
    print("   📷 순수 카메라 하드웨어 테스트")
    print("========================================")
    
    # 1. 윈도우 DSHOW 백엔드로 시도 (가장 호환성 좋음)
    print("\n[Attempt 1] cv2.CAP_DSHOW + 1280x720 + MJPG")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # 설정
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다 (Open Failed).")
        print("   -> 다른 프로그램(크롬, 줌 등)이 카메라를 쓰고 있는지 확인하세요.")
        return

    print("✅ 카메라 장치 연결 성공!")
    
    # 해상도 확인
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"   -> 설정된 해상도: {int(w)} x {int(h)}")

    print("\n🎥 화면을 띄웁니다. (종료하려면 화면 클릭 후 'Q')")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ 프레임 읽기 실패 (Black/Null Frame)")
            time.sleep(0.5)
            continue
            
        # 화면에 정보 표시
        cv2.putText(frame, f"Frame: {frame_count}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 화면이 너무 검은지 체크 (밝기 평균)
        brightness = frame.mean()
        if brightness < 10:
            cv2.putText(frame, f"DARK WARNING ({brightness:.1f})", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Pure Camera Test", frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
            
        frame_count += 1
        
    cap.release()
    cv2.destroyAllWindows()
    print("종료.")

if __name__ == "__main__":
    main()