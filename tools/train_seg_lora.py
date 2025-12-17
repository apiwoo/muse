# Project MUSE - train_seg_lora.py
# Entry Point for High-Precision Segmentation Training (MODNet + LoRA)
# Updated: Smart Skip (Skip training if .pth weights exist)
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
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--force", action='store_true', help="Force retrain even if weights exist")
    args = parser.parse_args()

    print("========================================================")
    print(f"   [TRACK B] High-Precision Seg (LoRA) Training: {args.profile}")
    print(f"   Target Epochs: {args.epochs}")
    print("========================================================")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(root_dir, "assets", "models", "personal", f"modnet_lora_weights_{args.profile}.pth")

    try:
        # Step 1: Train (Smart Skip)
        if os.path.exists(weights_path) and not args.force:
            print(f"\n>>> [SKIP] Found existing LoRA weights: {os.path.basename(weights_path)}")
            print("    Skipping training step and proceeding to merge/convert.")
        else:
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