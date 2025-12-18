# Project MUSE - config.py
# (C) 2025 MUSE Corp. All rights reserved.
# Role: Multi-Profile Config Manager (Full CRUD + Hotkey Persistence)
# V25.0: Extended parameters for High-Precision Pipeline

import os
import json
import glob
import shutil


class ProfileManager:
    def __init__(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.root_dir, "recorded_data", "personal_data")
        self.profiles = {}

        self.scan_profiles()

    def scan_profiles(self):
        """Scan recorded_data folder to build profile list."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        subdirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        subdirs.sort()

        if not subdirs:
            self.create_profile("default", camera_id=0)
            subdirs = ["default"]

        for idx, profile_name in enumerate(subdirs):
            self._load_single_profile(profile_name, idx)

    def _load_single_profile(self, profile_name, idx):
        config_path = os.path.join(self.data_dir, profile_name, "config.json")

        # [V25.0] Default Parameters - Extended for High-Precision Pipeline
        default_config = {
            "camera_id": 0,
            "hotkey": "",
            "params": {
                # ===== 기존 파라미터 (유지) =====
                'eye_scale': 0.0,
                'face_v': 0.0,
                'nose_slim': 0.0,
                'head_scale': 0.0,
                'shoulder_narrow': 0.0,
                'ribcage_slim': 0.0,
                'waist_slim': 0.0,
                'hip_widen': 0.0,
                'skin_smooth': 0.0,
                'skin_tone': 0.0,  # -1.0(Pale) ~ 1.0(Rosy)
                'show_body_debug': False,

                # ===== V25.0 신규 파라미터 =====
                # [Advanced Skin - Frequency Separation]
                'flatten_strength': 0.3,    # 피부 평탄화 강도 (0.0 ~ 1.0)
                'detail_preserve': 0.7,     # 고주파 디테일 보존량 (0.0 ~ 1.0)
                'gf_radius': 8,             # Guided Filter 반경 (4 ~ 16)
                'gf_epsilon': 0.04,         # Guided Filter 엣지 보존 (0.01 ~ 0.1)

                # [Color Grading]
                'color_temperature': 0.0,   # 색온도: -1.0(Cool/Blue) ~ 1.0(Warm/Yellow)
                'color_tint': 0.0,          # 틴트: -1.0(Green) ~ 1.0(Magenta)

            }
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded = json.load(f)
                    for k, v in loaded.items():
                        if k == "params":
                            # Merge loaded params with defaults (preserves new V25.0 params)
                            default_config["params"].update(v)
                        else:
                            default_config[k] = v
            except Exception as e:
                print(f"[WARNING] [{profile_name}] Config load failed: {e}")

        self.profiles[profile_name] = default_config

        if not os.path.exists(config_path):
            self.save_profile(profile_name, default_config)

    def create_profile(self, profile_name, camera_id=0, hotkey=""):
        target_dir = os.path.join(self.data_dir, profile_name)
        if os.path.exists(target_dir):
            print(f"[WARNING] Profile {profile_name} already exists.")
            return False

        os.makedirs(target_dir, exist_ok=True)
        self._load_single_profile(profile_name, 0)
        self.profiles[profile_name]['camera_id'] = camera_id
        self.profiles[profile_name]['hotkey'] = hotkey
        self.save_profile(profile_name, self.profiles[profile_name])
        return True

    def delete_profile(self, profile_name):
        if profile_name not in self.profiles:
            return False
        target_dir = os.path.join(self.data_dir, profile_name)
        try:
            shutil.rmtree(target_dir)
            del self.profiles[profile_name]
            return True
        except Exception as e:
            print(f"[ERROR] Failed to delete profile: {e}")
            return False

    def get_profile_list(self):
        return sorted(list(self.profiles.keys()))

    def get_config(self, profile_name):
        return self.profiles.get(profile_name, {})

    def get_profile_path(self, profile_name):
        return os.path.join(self.data_dir, profile_name)

    def update_params(self, profile_name, new_params):
        if profile_name in self.profiles:
            self.profiles[profile_name]['params'] = new_params
            self.save_profile(profile_name, self.profiles[profile_name])

    def update_camera_id(self, profile_name, cam_id):
        if profile_name in self.profiles:
            self.profiles[profile_name]['camera_id'] = cam_id
            self.save_profile(profile_name, self.profiles[profile_name])

    def update_hotkey(self, profile_name, hotkey_str):
        if profile_name in self.profiles:
            self.profiles[profile_name]['hotkey'] = hotkey_str
            self.save_profile(profile_name, self.profiles[profile_name])

    def save_profile(self, profile_name, config_data):
        path = os.path.join(self.data_dir, profile_name, "config.json")
        try:
            with open(path, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            print(f"[ERROR] Config save failed ({profile_name}): {e}")

    # =========================================================================
    # V25.0 Helper Methods
    # =========================================================================
    def get_v25_defaults(self):
        """Return V25.0 default parameter values"""
        return {
            'flatten_strength': 0.3,
            'detail_preserve': 0.7,
            'gf_radius': 8,
            'gf_epsilon': 0.04,
            'color_temperature': 0.0,
            'color_tint': 0.0
        }

    def upgrade_profile_to_v25(self, profile_name):
        """
        Upgrade existing profile to include V25.0 parameters
        (Non-destructive - only adds missing keys)
        """
        if profile_name not in self.profiles:
            return False

        defaults = self.get_v25_defaults()
        current_params = self.profiles[profile_name].get('params', {})

        updated = False
        for key, default_val in defaults.items():
            if key not in current_params:
                current_params[key] = default_val
                updated = True

        if updated:
            self.profiles[profile_name]['params'] = current_params
            self.save_profile(profile_name, self.profiles[profile_name])
            print(f"[CONFIG] Profile '{profile_name}' upgraded to V25.0")

        return updated

    def upgrade_all_profiles_to_v25(self):
        """Upgrade all profiles to V25.0 format"""
        upgraded_count = 0
        for profile_name in self.profiles.keys():
            if self.upgrade_profile_to_v25(profile_name):
                upgraded_count += 1
        if upgraded_count > 0:
            print(f"[CONFIG] Upgraded {upgraded_count} profiles to V25.0")
        return upgraded_count
