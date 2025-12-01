# Project MUSE - recorder.py
# Data Acquisition Tool for Personalized AI
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import os
import time
import sys
import numpy as np

class DataRecorder:
    def __init__(self, output_dir="recorded_data"):
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(0) # 0번 카메라
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 저장 경로 생성
        os.makedirs(output_dir, exist_ok=True)
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.clean_plate = None
        self.is_recording = False
        self.video_writer = None
        
        print(f"📂 [Recorder] 세션 시작: {self.session_dir}")

    def run(self):
        print("========================================================")
        print("   MUSE Data Recorder (Personalization Step 1)")
        print("========================================================")
        print("   [Step 1] 빈 방(배경) 찍기")
        print("     - 화면 밖으로 나가세요.")
        print("     - 'B' 키를 누르면 배경(Clean Plate)이 저장됩니다.")
        print("   [Step 2] 데이터 녹화")
        print("     - 화면 안으로 들어오세요.")
        print("     - 'R' 키를 누르면 녹화가 시작됩니다.")
        print("     - 다양한 동작(팔 벌리기, 앉기, 돌기)을 수행하세요.")
        print("     - 다시 'R'을 누르면 녹화가 저장됩니다.")
        print("   [Quit] 'Q' 종료")
        print("========================================================")

        while True:
            ret, frame = self.cap.read()
            if not ret: break

            display = frame.copy()
            h, w = display.shape[:2]

            # 상태 표시 UI
            status_text = "Ready"
            color = (0, 255, 0)

            if self.clean_plate is None:
                cv2.putText(display, "STEP 1: Move out & Press 'B' for Background", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif not self.is_recording:
                cv2.putText(display, "STEP 2: Press 'R' to Start Recording", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # 우측 상단에 캡처된 배경 작게 보여주기
                small_bg = cv2.resize(self.clean_plate, (320, 180))
                display[0:180, w-320:w] = small_bg
                cv2.rectangle(display, (w-320, 0), (w, 180), (255, 255, 0), 2)
            else:
                cv2.putText(display, "🔴 RECORDING... (Press 'R' to Stop)", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(display, (30, 40), 10, (0, 0, 255), -1)

            cv2.imshow("MUSE Recorder", display)
            
            key = cv2.waitKey(1) & 0xFF

            # [Key Logic]
            if key == ord('q'):
                break
            
            elif key == ord('b'): # Background Capture
                self.clean_plate = frame.copy()
                path = os.path.join(self.session_dir, "background.jpg")
                cv2.imwrite(path, self.clean_plate)
                print(f"📸 배경 저장 완료: {path}")
            
            elif key == ord('r'): # Record Toggle
                if not self.is_recording:
                    # 녹화 시작
                    self.is_recording = True
                    video_path = os.path.join(self.session_dir, "train_video.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
                    print(f"🎥 녹화 시작: {video_path}")
                else:
                    # 녹화 종료
                    self.is_recording = False
                    if self.video_writer:
                        self.video_writer.release()
                        self.video_writer = None
                    print("💾 녹화 저장 완료.")
            
            # 녹화 중일 때 프레임 저장
            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)

        self.cleanup()

    def cleanup(self):
        if self.cap: self.cap.release()
        if self.video_writer: self.video_writer.release()
        cv2.destroyAllWindows()
        print("👋 레코더 종료.")

if __name__ == "__main__":
    recorder = DataRecorder()
    recorder.run()