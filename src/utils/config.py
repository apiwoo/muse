# Project MUSE - config.py
# (C) 2025 MUSE Corp. All rights reserved.
# 역할: 멀티 프로파일 설정 관리 (카메라 매핑, 보정값 저장/로드)

import os
import json
import glob

class ProfileManager:
    def __init__(self):
        # 프로젝트 루트 경로 계산
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.root_dir, "recorded_data", "personal_data")
        self.profiles = {} # name -> {camera_id, params}
        
        # 초기화 시 스캔
        self.scan_profiles()

    def scan_profiles(self):
        """recorded_data 폴더를 스캔하여 프로파일 목록을 구축합니다."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            
        # 디렉토리만 찾기
        subdirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        
        # 알파벳순 정렬 (front, top, side...) -> 이 순서대로 1, 2, 3번 키 할당
        subdirs.sort()
        
        if not subdirs:
            # 프로파일이 하나도 없으면 'default' 생성
            subdirs = ["default"]
            os.makedirs(os.path.join(self.data_dir, "default"), exist_ok=True)
        
        print(f"📂 [ProfileManager] 스캔된 프로파일: {subdirs}")

        for idx, profile_name in enumerate(subdirs):
            config_path = os.path.join(self.data_dir, profile_name, "config.json")
            
            # 기본 템플릿
            default_config = {
                "camera_id": idx, # 기본값: 폴더 순서대로 0, 1, 2번 카메라 할당
                "params": {
                    'eye_scale': 0.0,
                    'face_v': 0.0,
                    'head_scale': 0.0,
                    'shoulder_narrow': 0.0,
                    'ribcage_slim': 0.0,
                    'waist_slim': 0.0,
                    'hip_widen': 0.0,
                    'show_body_debug': False
                }
            }
            
            # 기존 설정 파일이 있으면 로드 및 병합
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        loaded = json.load(f)
                        # 최상위 키 병합
                        for k, v in loaded.items():
                            if k == "params":
                                default_config["params"].update(v)
                            else:
                                default_config[k] = v
                except Exception as e:
                    print(f"⚠️ [{profile_name}] 설정 로드 실패, 기본값 사용: {e}")
            
            # 메모리에 등록
            self.profiles[profile_name] = default_config
            
            # 파일 갱신 (누락된 키 추가 저장)
            self.save_profile(profile_name, default_config)

    def get_profile_list(self):
        """정렬된 프로파일 이름 리스트 반환"""
        return sorted(list(self.profiles.keys()))

    def get_config(self, profile_name):
        return self.profiles.get(profile_name, {})

    def update_params(self, profile_name, new_params):
        """특정 프로파일의 파라미터만 업데이트하고 저장"""
        if profile_name in self.profiles:
            self.profiles[profile_name]['params'] = new_params
            self.save_profile(profile_name, self.profiles[profile_name])

    def save_profile(self, profile_name, config_data):
        path = os.path.join(self.data_dir, profile_name, "config.json")
        try:
            with open(path, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            print(f"❌ 설정 저장 실패 ({profile_name}): {e}")