# Project MUSE - train_seg_lora.py
# Entry Point for High-Precision Segmentation Training (MODNet + LoRA)
# Updated: Added --epochs argument for longer training
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
    # [Update] Epochs argument added (Default 50 for better convergence)
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()

    print("========================================================")
    print(f"   [TRACK B] High-Precision Seg (LoRA) Training: {args.profile}")
    print(f"   Target Epochs: {args.epochs}")
    print("========================================================")

    try:
        # Step 1: Train
        print(f"\n>>> Step 1: MODNet LoRA Fine-Tuning ({args.epochs} Epochs)")
        trainer = LoRATrainer(args.session, args.profile, task='seg', epochs=args.epochs, batch_size=8)
        trainer.train()
        
        # Step 2: Merge & Convert
        print("\n>>> Step 2: Merge & Engine Build")
        success = merge_and_convert(args.profile, mode='seg')
        
        if success:
            print("\n[DONE] Seg LoRA Pipeline Complete.")
        else:
            print("\n[FAIL] Engine generation failed.")
            sys.exit(1)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()