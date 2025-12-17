# Project MUSE - trainer_lora.py
# High-Precision LoRA Trainer (Universal: Seg & Pose)
# Updated: Support for MODNet (Seg) and ViTPose (Pose)
# Updated v1.2: Fatal Error on Base Model Load Failure
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

# Paths
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

from src.ai.models.vitpose_lora import ViTPoseLoRA
from src.ai.models.modnet_lora import MODNetLoRA
from src.ai.distillation.trainer import MuseDataset # Reuse Dataset

class LoRATrainer:
    def __init__(self, session_name, target_profile, task='pose', epochs=20, batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.profile = target_profile
        self.task = task # 'seg' or 'pose'
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.data_dir = os.path.join(root_dir, "recorded_data", session_name, target_profile)
        self.model_save_dir = os.path.join(root_dir, "assets", "models", "personal")
        
        # Model Paths
        self.pose_base_path = os.path.join(root_dir, "assets", "models", "tracking", "vitpose_base_coco_256x192.pth")
        
        # MODNet Base: Prioritize CKPT, fallback to others if needed.
        # For simplicity, we assume download_models.py fetched 'modnet_webcam_portrait_matting.ckpt'
        self.seg_base_path = os.path.join(root_dir, "assets", "models", "segmentation", "modnet_webcam_portrait_matting.ckpt")
        
        # Config resolution
        if self.task == 'pose':
            self.input_size = (192, 256) # W, H
        else:
            self.input_size = (512, 512) # MODNet Training Resolution (Square)

        print(f"[LoRA] Trainer Init: {self.task.upper()} for '{self.profile}'")

    def train(self):
        # 1. Dataset
        dataset = MuseDataset(self.data_dir, input_size=self.input_size)
        if len(dataset) == 0:
            print("[ERROR] No data found.")
            return
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # 2. Model Init
        if self.task == 'pose':
            model = self._init_pose_model()
            loss_fn = nn.MSELoss(reduction='none')
        else:
            model = self._init_seg_model()
            loss_fn = nn.MSELoss() # Matte Regression Loss

        model.to(self.device)

        # 3. Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        # 4. Training Loop
        model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            total_raw_loss = 0.0 # [New] 순수 MSE 기록용 (Pose)
            
            pbar = tqdm(dataloader, desc=f"LoRA ({self.task.upper()}) Ep {epoch+1}/{self.epochs}", leave=False)
            
            for imgs, masks, heatmaps in pbar:
                imgs = imgs.to(self.device)
                
                optimizer.zero_grad()
                
                if self.task == 'pose':
                    heatmaps = heatmaps.to(self.device)
                    preds = model(imgs)
                    
                    # Weighted MSE for Pose (Shoulder/Hip focus)
                    weights = torch.ones(17, device=self.device)
                    weights[[5, 6, 11, 12]] = 10.0
                    
                    # [Detailed Loss]
                    loss_map = loss_fn(preds, heatmaps) # (B, 17, H, W)
                    
                    # 1. Raw MSE (Global average, unweighted) -> 전체적인 수렴도 확인
                    raw_loss = loss_map.mean()
                    
                    # 2. Weighted Loss (Optimization Target) -> 중요 부위 집중
                    loss = (loss_map.mean(dim=(2,3)) * weights).mean()
                    
                    total_raw_loss += raw_loss.item()
                    
                    # Log both
                    pbar.set_postfix_str(f"L:{loss.item():.6f} (Raw:{raw_loss.item():.6f})")
                    
                else: # Seg
                    masks = masks.to(self.device) # (B, 1, H, W)
                    preds = model(imgs) # (B, 1, H, W)
                    
                    # Simple MSE on Alpha Matte
                    loss = loss_fn(preds, masks)
                    
                    # Log detailed
                    pbar.set_postfix_str(f"Loss:{loss.item():.6f}")

                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            
            # [Epoch Summary]
            avg_loss = total_loss / len(dataloader)
            
            if self.task == 'pose':
                avg_raw = total_raw_loss / len(dataloader)
                print(f"   Epoch {epoch+1}: Weighted {avg_loss:.6f} | Raw MSE {avg_raw:.6f}")
            else:
                print(f"   Epoch {epoch+1}: Avg Loss {avg_loss:.6f}")

        # 5. Save Weights
        self._save_weights(model)

    def _init_pose_model(self):
        if not os.path.exists(self.pose_base_path):
            print(f"[CRITICAL] Base ViTPose not found: {self.pose_base_path}")
            print("   -> Run 'tools/download_models.py' first.")
            sys.exit(1)
            
        model = ViTPoseLoRA(img_size=(256, 192), patch_size=16, embed_dim=768, depth=12)
        try:
            checkpoint = torch.load(self.pose_base_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            # Remap
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('backbone.', '')
                if 'keypoint_head.deconv_layers.0.' in new_k: new_k = new_k.replace('keypoint_head.deconv_layers.0.', 'keypoint_head.0.')
                elif 'keypoint_head.deconv_layers.1.' in new_k: new_k = new_k.replace('keypoint_head.deconv_layers.1.', 'keypoint_head.1.')
                elif 'keypoint_head.deconv_layers.3.' in new_k: new_k = new_k.replace('keypoint_head.deconv_layers.3.', 'keypoint_head.3.')
                elif 'keypoint_head.deconv_layers.4.' in new_k: new_k = new_k.replace('keypoint_head.deconv_layers.4.', 'keypoint_head.4.')
                elif 'keypoint_head.final_layer.' in new_k: new_k = new_k.replace('keypoint_head.final_layer.', 'keypoint_head.6.')
                new_state_dict[new_k] = v
                
            model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print(f"[CRITICAL] Failed to load ViTPose weights: {e}")
            sys.exit(1)

        model.inject_lora(rank=8)
        return model

    def _init_seg_model(self):
        # NOTE: MODNet weights loading logic needs to be robust
        model = MODNetLoRA(in_channels=3)
        
        if os.path.exists(self.seg_base_path):
            print(f"   -> Loading MODNet Base: {os.path.basename(self.seg_base_path)}")
            # Handle CKPT loading (state_dict might be nested)
            try:
                checkpoint = torch.load(self.seg_base_path, map_location='cpu')
                state_dict = checkpoint.get('state_dict', checkpoint)
                
                # Cleanup keys if trained with DataParallel or Lightning
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_k = k.replace('module.', '')
                    new_state_dict[new_k] = v
                    
                model.load_state_dict(new_state_dict, strict=False)
            except Exception as e:
                print(f"   [CRITICAL] Failed to load MODNet weights: {e}")
                print("   -> Your file might be corrupted (HTML downloaded instead of Model).")
                print("   -> Please delete the file and run 'tools/download_models.py'.")
                sys.exit(1) # Stop execution prevents garbage training
        else:
            print("   [CRITICAL] MODNet Base weights not found.")
            print("   -> Please run 'tools/download_models.py'.")
            sys.exit(1)

        model.inject_lora(rank=4)
        return model

    def _save_weights(self, model):
        prefix = "vitpose" if self.task == 'pose' else "modnet"
        save_path = os.path.join(self.model_save_dir, f"{prefix}_lora_weights_{self.profile}.pth")
        
        # Extract LoRA + Head (for Pose) or LoRA (for Seg)
        save_dict = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
        
        if self.task == 'pose':
            head_dict = {k: v for k, v in model.state_dict().items() if 'keypoint_head' in k}
            save_dict.update(head_dict)
            
        torch.save(save_dict, save_path)
        print(f"[DONE] Saved LoRA Weights: {os.path.basename(save_path)}")

if __name__ == "__main__":
    pass