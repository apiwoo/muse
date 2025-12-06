# Project MUSE - train_student.py
# (C) 2025 MUSE Corp.
# Supports --task and --profile for targeted training

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