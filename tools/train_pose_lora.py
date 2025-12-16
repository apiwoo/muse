# Project MUSE - train_pose_lora.py
# Orchestrator for High-Precision LoRA Track
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import argparse
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ai.distillation.trainer_lora import LoRATrainer
from tools.merge_lora_to_engine import merge_and_convert

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", default="personal_data", nargs="?")
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()

    print("========================================================")
    print(f"   [TRACK B] High-Precision LoRA Training: {args.profile}")
    print("========================================================")

    try:
        # Step 1: Train
        print("\n>>> Step 1: LoRA Fine-Tuning")
        trainer = LoRATrainer(args.session, args.profile, epochs=20, batch_size=8)
        trainer.train()
        
        # Step 2: Merge & Convert
        print("\n>>> Step 2: Merge & Engine Build")
        success = merge_and_convert(args.profile)
        
        if success:
            print("\n[DONE] All steps completed successfully.")
            print("To use this mode, select 'High-Precision' in Launcher.")
        else:
            print("\n[FAIL] Engine generation failed.")
            sys.exit(1)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()