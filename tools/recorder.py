# Project MUSE - recorder.py
# Data Acquisition Tool for Multi-Camera Personalization (Profile Support)
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import os
import time
import sys
import shutil
import glob
import numpy as np

# [Optional] Windows Camera Name Detection
try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False
    print("\n⚠️ [경고] 'pygrabber' 모듈이 설치되지 않았습니다.")
    print("   -> 현재 카메라가 'Camera Device 0'과 같이 숫자로만 표시됩니다.")
    print("   -> 실제 이름(예: Logitech C920)을 보려면 'pip install pygrabber'를 설치하세요.")

class DataRecorder:
    def __init__(self, output_dir="recorded_data"):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.root_dir, output_dir)
        
        # 데이터 루트 (이 안에 front, top 등 프로파일 폴더가 생성됨)
        self.root_data_path = os.path.join(self.output_dir, "personal_data")
        
        # 모델 경로 (백업용)
        self.model_dir = os.path.join(self.root_dir, "assets", "models", "personal")

        self.cap = None
        self.clean_plate = None
        self.is_recording = False
        self.video_writer = None
        self.record_count = 0
        self.current_profile = "default" # 현재 작업 중인 프로파일 이름
        self.profile_dir = ""            # 현재 프로파일 경로
        
        # [Time Tracking]
        self.total_recorded_time = 0.0
        self.current_start_time = 0.0
        
        # [Step 1] 전체 모드 선택 (초기화 여부)
        self._select_global_mode()
        
        # [Step 2] 프로파일 및 카메라 설정 루프
        self._setup_profile_session()

    def _select_global_mode(self):
        print("\n========================================================")
        print("   MUSE Multi-Cam Recorder - Global Mode")
        print("========================================================")
        print("   1. [RESET ALL] : 이사/환경 변화 (모든 프로파일 삭제 & 초기화)")
        print("   2. [MANAGE]    : 프로파일 추가/수정 (기존 데이터 유지)")
        print("========================================================")
        
        while True:
            choice = input("👉 선택 (1 or 2): ").strip()
            if choice == '1':
                self._reset_all_data()
                break
            elif choice == '2':
                if not os.path.exists(self.root_data_path):
                    print("⚠️ 기존 데이터가 없습니다. 자동으로 폴더를 생성합니다.")
                    os.makedirs(self.root_data_path, exist_ok=True)
                break
            else:
                print("❌ 잘못된 입력입니다.")

    def _reset_all_data(self):
        print("\n🧹 [RESET ALL] 전체 데이터 초기화 및 백업...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_root = os.path.join(self.output_dir, "backup", f"backup_{timestamp}")
        
        # 데이터 백업
        if os.path.exists(self.root_data_path):
            try:
                os.makedirs(backup_root, exist_ok=True)
                shutil.move(self.root_data_path, os.path.join(backup_root, "personal_data"))
                print(f"   ✅ 데이터 백업 완료: {backup_root}")
            except Exception as e:
                print(f"   ⚠️ 데이터 백업 실패: {e}")
        
        # 모델 백업
        if os.path.exists(self.model_dir):
            try:
                model_backup = os.path.join(backup_root, "models")
                os.makedirs(model_backup, exist_ok=True)
                for ext in ["*.pth", "*.engine"]:
                    for f in glob.glob(os.path.join(self.model_dir, ext)):
                        shutil.move(f, model_backup)
                print(f"   ✅ 모델 파일 백업 완료")
            except Exception as e: pass

        os.makedirs(self.root_data_path, exist_ok=True)
        print("✨ 초기화 완료.")

    def _get_camera_list(self):
        """시스템에 연결된 카메라 목록을 반환합니다."""
        cameras = []
        if HAS_PYGRABBER:
            try:
                graph = FilterGraph()
                devices = graph.get_input_devices()
                for i, name in enumerate(devices):
                    cameras.append((i, name))
            except Exception as e:
                print(f"⚠️ 카메라 이름 조회 실패: {e}")
        
        # pygrabber가 없거나 실패한 경우, 단순 ID 스캔 (0~5번 시도)
        if not cameras:
            if not HAS_PYGRABBER:
                print("\n💡 [Info] 카메라 이름이 안 보이나요? 'pip install pygrabber'를 해보세요.")
                
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cameras.append((i, f"Camera Device {i}"))
                    cap.release()
        
        return cameras

    def _setup_profile_session(self):
        """특정 프로파일(front, top 등)을 선택하고 카메라를 연결합니다."""
        print("\n========================================================")
        print("   Profile & Camera Setup")
        print("========================================================")
        
        # 현재 존재하는 프로파일 목록 표시
        existing_profiles = [d for d in os.listdir(self.root_data_path) if os.path.isdir(os.path.join(self.root_data_path, d))]
        if existing_profiles:
            print(f"📂 기존 프로파일: {', '.join(existing_profiles)}")
        
        # 1. 프로파일 이름 입력
        while True:
            p_name = input("👉 프로파일 이름 입력 (예: front, top, side): ").strip()
            if p_name:
                self.current_profile = p_name
                self.profile_dir = os.path.join(self.root_data_path, p_name)
                os.makedirs(self.profile_dir, exist_ok=True)
                print(f"   -> 타겟 폴더: {self.profile_dir}")
                break
        
        # 2. 카메라 선택 (이름 표시)
        cameras = self._get_camera_list()
        print(f"\n🔍 감지된 카메라 목록:")
        for idx, name in cameras:
            print(f"   [{idx}] {name}")
            
        while True:
            try:
                cam_id_str = input("👉 사용할 카메라 ID 입력: ").strip()
                cam_id = int(cam_id_str)
                
                # 유효한 ID인지 확인 (목록에 없어도 강제 입력 가능하게 함 - 고급 사용자용)
                valid_ids = [c[0] for c in cameras]
                if cam_id not in valid_ids:
                    print(f"⚠️ 경고: 감지된 목록에 없는 ID({cam_id})입니다.")
                
                print(f"   📷 카메라({cam_id}) 연결 시도...")
                if self.cap: self.cap.release()
                self.cap = cv2.VideoCapture(cam_id)
                
                # 해상도 설정 (FHD 권장)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                if not self.cap.isOpened():
                    print("   ❌ 카메라를 열 수 없습니다. 다른 ID를 입력하세요.")
                    continue
                
                # 테스트 리딩
                ret, _ = self.cap.read()
                if ret:
                    print("   ✅ 카메라 연결 성공!")
                    break
                else:
                    print("   ⚠️ 카메라는 열렸으나 화면이 안 나옵니다.")
            except ValueError:
                print("   ❌ 숫자를 입력하세요.")

        # 3. 기존 데이터 상태 확인 (Fine-tune 모드일 경우)
        self._load_existing_status()

    def _load_existing_status(self):
        # 배경 확인
        bg_path = os.path.join(self.profile_dir, "background.jpg")
        if os.path.exists(bg_path):
            self.clean_plate = cv2.imread(bg_path)
            print("   ✅ 기존 배경(background.jpg) 로드됨")
        else:
            self.clean_plate = None
            print("   ⚠️ 배경 촬영이 필요합니다 ('B' 키).")

        # 영상 인덱스 및 시간 확인
        files = glob.glob(os.path.join(self.profile_dir, "train_video_*.mp4"))
        max_idx = 0
        total_seconds = 0.0
        
        for f in files:
            try:
                name = os.path.splitext(os.path.basename(f))[0]
                idx = int(name.replace("train_video_", ""))
                if idx > max_idx: max_idx = idx
            except: pass
            
            # 시간 계산 (대략적)
            cap = cv2.VideoCapture(f)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps > 0: total_seconds += (frames / fps)
            cap.release()
            
        self.record_count = max_idx
        self.total_recorded_time = total_seconds
        print(f"   📊 [{self.current_profile}] 기존 영상: {len(files)}개 ({self._fmt_time(total_seconds)})")

    def _fmt_time(self, seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def run(self):
        print("\n========================================================")
        print(f"   MUSE Recorder - [{self.current_profile}] Mode")
        print("========================================================")
        print("   [Key Controls]")
        print("   - 'B': 배경 촬영 (빈 방)")
        print("   - 'R': 녹화 시작/중지")
        print("   - 'N': 새로운 프로파일로 전환 (Next Profile)")
        print("   - 'Q': 종료")
        print("========================================================")

        while True:
            if not self.cap or not self.cap.isOpened():
                print("❌ 카메라 연결 끊김")
                break

            ret, frame = self.cap.read()
            if not ret: break

            display = frame.copy()
            h, w = display.shape[:2]

            # Time Calc
            current_clip_time = 0.0
            if self.is_recording:
                current_clip_time = time.time() - self.current_start_time
            total_display_time = self.total_recorded_time + current_clip_time

            # UI
            ui_color = (0, 255, 0)
            
            # Profile Name
            cv2.putText(display, f"Profile: {self.current_profile}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if self.clean_plate is None:
                cv2.putText(display, "STEP 1: Press 'B' (Clean Plate)", (30, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif not self.is_recording:
                cv2.putText(display, "READY: Press 'R'", (30, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 배경 썸네일
                small_bg = cv2.resize(self.clean_plate, (320, 180))
                display[0:180, w-320:w] = small_bg
                cv2.rectangle(display, (w-320, 0), (w, 180), (255, 255, 0), 2)
            else:
                cv2.putText(display, "🔴 RECORDING...", (30, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(display, (20, 90), 10, (0, 0, 255), -1)

            # Timer
            cv2.putText(display, f"Current: {self._fmt_time(current_clip_time)}", (w - 300, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(display, f"Total  : {self._fmt_time(total_display_time)}", (w - 300, 260), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("MUSE Recorder", display)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            
            # [N]ext Profile: 다른 카메라/앵글 설정으로 이동
            elif key == ord('n'):
                if self.is_recording:
                    print("⚠️ 녹화 중에는 전환할 수 없습니다.")
                else:
                    print("\n🔄 다른 프로파일로 전환합니다...")
                    self._setup_profile_session() # 재설정 진입
            
            elif key == ord('b'):
                self.clean_plate = frame.copy()
                path = os.path.join(self.profile_dir, "background.jpg")
                cv2.imwrite(path, self.clean_plate)
                print(f"📸 [{self.current_profile}] 배경 저장 완료")
            
            elif key == ord('r'):
                if self.clean_plate is None:
                    print("⚠️ 배경을 먼저 찍어주세요 ('B')")
                    continue

                if not self.is_recording:
                    # Start
                    self.is_recording = True
                    self.current_start_time = time.time()
                    self.record_count += 1
                    
                    filename = f"train_video_{self.record_count:02d}.mp4"
                    video_path = os.path.join(self.profile_dir, filename)
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
                    print(f"🎥 녹화 시작: {filename}")
                else:
                    # Stop
                    self.is_recording = False
                    elapsed = time.time() - self.current_start_time
                    self.total_recorded_time += elapsed
                    
                    if self.video_writer:
                        self.video_writer.release()
                        self.video_writer = None
                    print(f"💾 저장 완료. Total: {self._fmt_time(self.total_recorded_time)}")

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