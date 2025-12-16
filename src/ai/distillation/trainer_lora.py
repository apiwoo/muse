# Project MUSE - trainer_lora.py
# High-Precision LoRA Trainer for ViTPose
# Focus: Waist/Hip/Shoulder Precision
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.ai.models.vitpose_lora import ViTPoseLoRA
from src.ai.distillation.trainer import MuseDataset # Reuse Dataset

class LoRATrainer:
    def __init__(self, session_name, target_profile, epochs=20, batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.profile = target_profile
        self.epochs = epochs
        self.batch_size = batch_size
        
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.data_dir = os.path.join(base_path, "recorded_data", session_name, target_profile)
        self.model_save_dir = os.path.join(base_path, "assets", "models", "personal")
        self.base_model_path = os.path.join(base_path, "assets", "models", "tracking", "vitpose_base_coco_256x192.pth")
        
        if not os.path.exists(self.base_model_path):
            raise FileNotFoundError("Base ViTPose model not found. Run download_models.py.")

        # Input size for ViTPose-Base
        self.input_size = (192, 256) # W, H

    def train(self):
        print(f"[LoRA] Starting High-Precision Training for '{self.profile}'...")
        
        # 1. Dataset
        dataset = MuseDataset(self.data_dir, input_size=self.input_size)
        if len(dataset) == 0:
            print("[ERROR] No data found.")
            return
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # 2. Model Init
        model = ViTPoseLoRA(img_size=(256, 192), patch_size=16, embed_dim=768, depth=12).to(self.device)
        
        # Load Base Weights
        print(f"   -> Loading Base Weights: {os.path.basename(self.base_model_path)}")
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Remap weights (similar to trt_converter)
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
        
        # Inject LoRA
        model.inject_lora(rank=8)
        model.to(self.device)

        # 3. Optimizer (Only train requires_grad params)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        # 4. Weighted Loss
        # 5,6: Shoulders, 11,12: Hips
        weights = torch.ones(17, device=self.device)
        weights[[5, 6, 11, 12]] = 10.0
        mse_loss = nn.MSELoss(reduction='none')

        # 5. Training Loop
        model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"LoRA Ep {epoch+1}/{self.epochs}", leave=False)
            
            for imgs, _, heatmaps in pbar:
                imgs = imgs.to(self.device)
                heatmaps = heatmaps.to(self.device)
                
                optimizer.zero_grad()
                
                preds = model(imgs)
                
                # Weighted MSE
                loss_map = mse_loss(preds, heatmaps) # (B, 17, H, W)
                weighted_loss = (loss_map.mean(dim=(2,3)) * weights).mean()
                
                weighted_loss.backward()
                optimizer.step()
                
                total_loss += weighted_loss.item()
                pbar.set_postfix_str(f"Loss: {weighted_loss.item():.4f}")
            
            scheduler.step()
            print(f"   Epoch {epoch+1}: Avg Loss {total_loss/len(dataloader):.4f}")

        # 6. Save LoRA Weights Only (Lightweight)
        save_path = os.path.join(self.model_save_dir, f"vitpose_lora_weights_{self.profile}.pth")
        
        lora_dict = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
        # Also save head as it might adapt slightly
        head_dict = {k: v for k, v in model.state_dict().items() if 'keypoint_head' in k}
        
        final_dict = {**lora_dict, **head_dict}
        torch.save(final_dict, save_path)
        print(f"[DONE] LoRA Weights Saved: {os.path.basename(save_path)}")

if __name__ == "__main__":
    pass