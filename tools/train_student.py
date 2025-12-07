# Project MUSE - train_student.py
# (C) 2025 MUSE Corp.
# Supports --task and --profile for targeted training
# Updated: Skip if model exists (Smart Resume)

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ai.distillation.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train Student Model")
    parser.add_argument("session", nargs='?', default="personal_data", help="Session folder name")
    parser.add_argument("--task", choices=['seg', 'pose'], default='seg', help="Training task (seg or pose)")
    parser.add_argument("--profile", type=str, default=None, help="Specific profile to train (e.g., front)")
    args = parser.parse_args()

    session_name = args.session
    task = args.task
    target_profile = args.profile
    
    print("========================================================")
    print(f"   [START] Training Student: {session_name} ({task.upper()})")
    if target_profile:
        print(f"   [TARGET] Profile: {target_profile}")
    print("========================================================")
    
    # [Smart Resume Logic]
    # Check if the output model already exists.
    # Expected Path: assets/models/personal/student_{task}_{profile}.pth
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_save_dir = os.path.join(base_path, "assets", "models", "personal")
    
    if target_profile:
        save_name = f"student_{task}_{target_profile}.pth"
        save_path = os.path.join(model_save_dir, save_name)
        
        if os.path.exists(save_path):
            print(f"\n[SKIP] Model already exists: {save_name}")
            print("   -> To retrain, delete this file or use 'Reset' mode.")
            return

    epochs = 50 
    
    trainer = Trainer(session_name, task=task, target_profile=target_profile, epochs=epochs, batch_size=8)
    
    try:
        trainer.train_all_profiles()
    except KeyboardInterrupt:
        print("\n[STOP] Training interrupted.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()