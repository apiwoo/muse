# Project MUSE - config.py
# (C) 2025 MUSE Corp. All rights reserved.
# Role: Multi-Profile Config Manager

import os
import json
import glob

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
            subdirs = ["default"]
            os.makedirs(os.path.join(self.data_dir, "default"), exist_ok=True)
        
        print(f"[DIR] [ProfileManager] Scanned Profiles: {subdirs}")

        for idx, profile_name in enumerate(subdirs):
            config_path = os.path.join(self.data_dir, profile_name, "config.json")
            
            default_config = {
                "camera_id": idx, 
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
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        loaded = json.load(f)
                        for k, v in loaded.items():
                            if k == "params":
                                default_config["params"].update(v)
                            else:
                                default_config[k] = v
                except Exception as e:
                    print(f"[WARNING] [{profile_name}] Config load failed, using default: {e}")
            
            self.profiles[profile_name] = default_config
            self.save_profile(profile_name, default_config)

    def get_profile_list(self):
        return sorted(list(self.profiles.keys()))

    def get_config(self, profile_name):
        return self.profiles.get(profile_name, {})

    def update_params(self, profile_name, new_params):
        if profile_name in self.profiles:
            self.profiles[profile_name]['params'] = new_params
            self.save_profile(profile_name, self.profiles[profile_name])

    def save_profile(self, profile_name, config_data):
        path = os.path.join(self.data_dir, profile_name, "config.json")
        try:
            with open(path, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            print(f"[ERROR] Config save failed ({profile_name}): {e}")