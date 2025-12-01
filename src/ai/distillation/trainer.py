# Project MUSE - trainer.py
# The Knowledge Distillation Engine (Trainer)
# (C) 2025 MUSE Corp. All rights reserved.

import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# [MUSE Modules]
from src.ai.distillation.student.model_arch import MuseStudentModel

class MuseDataset(Dataset):
    def __init__(self, data_dir, input_size=(512, 512), heatmap_size=(128, 128)):
        self.data_dir = data_dir
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.label_dir = os.path.join(data_dir, "labels")
        
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        print(f"📂 [Dataset] Found {len(self.img_files)} samples in {data_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 1. Load Files
        img_path = self.img_files[idx]
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{basename}.png")
        label_path = os.path.join(self.label_dir, f"{basename}.json")

        # Load Image (BGR -> RGB)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]

        # Load Mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

        # Load Keypoints
        keypoints = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                data = json.load(f)
                keypoints = np.array(data['keypoints']) # [[x,y,c], ...]
        
        # 2. Resize & Normalize (Simple Scaling)
        # Training Resolution: 512x512 (Why? Trade-off between speed and detail)
        img_resized = cv2.resize(img, self.input_size)
        mask_resized = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
        
        # Image Normalization (0~1, then Standardize)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Mask Tensor (0 or 1)
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0) / 255.0
        mask_tensor = (mask_tensor > 0.5).float() # Binarize

        # 3. Generate Heatmaps for Pose (1/4 Scale)
        heatmaps = np.zeros((17, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        
        if len(keypoints) > 0:
            scale_x = self.heatmap_size[0] / w_orig
            scale_y = self.heatmap_size[1] / h_orig
            
            for k_idx, (x, y, conf) in enumerate(keypoints):
                if conf > 0.4: # Only valid points
                    # Scale coordinates
                    hx, hy = int(x * scale_x), int(y * scale_y)
                    self._add_gaussian(heatmaps[k_idx], hx, hy)

        heatmap_tensor = torch.from_numpy(heatmaps)

        return img_tensor, mask_tensor, heatmap_tensor

    def _add_gaussian(self, heatmap, x, y, sigma=2):
        h, w = heatmap.shape
        tmp_size = sigma * 3
        mu_x = int(x + 0.5)
        mu_y = int(y + 0.5)
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        
        if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
            return heatmap
            
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        
        g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
        img_x = max(0, ul[0]), min(br[0], w)
        img_y = max(0, ul[1]), min(br[1], h)
        
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        )
        return heatmap

class Trainer:
    def __init__(self, session_name, epochs=50, batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 [Trainer] Device: {self.device}")
        
        # Paths
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.data_dir = os.path.join(self.root_dir, "recorded_data", session_name)
        self.model_save_dir = os.path.join(self.root_dir, "assets", "models", "personal")
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Data
        self.dataset = MuseDataset(self.data_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        
        # Model
        self.model = MuseStudentModel(num_keypoints=17).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Losses
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
        self.epochs = epochs

    def train(self):
        print(f"🔥 [Start Training] Epochs: {self.epochs}")
        
        scaler = torch.cuda.amp.GradScaler() # Mixed Precision for speed
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for imgs, masks, heatmaps in pbar:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                heatmaps = heatmaps.to(self.device)
                
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    pred_seg, pred_pose = self.model(imgs)
                    
                    # 1. Segmentation Loss
                    # Resize pred_seg to 512x512 (Original) if needed, but it's already upsampled in model
                    # But if we used U-Net style, check output size.
                    # Our model outputs full res (512x512) for seg.
                    loss_seg = self.bce_loss(pred_seg, masks)
                    
                    # 2. Pose Loss
                    # pred_pose is 128x128 (1/4 scale). heatmaps is 128x128.
                    loss_pose = self.mse_loss(pred_pose, heatmaps) * 1000 # Scaling factor for balance
                    
                    total_loss = loss_seg + loss_pose

                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                running_loss += total_loss.item()
                pbar.set_postfix(loss=total_loss.item(), seg=loss_seg.item(), pose=loss_pose.item())
            
            self.scheduler.step()
            
            # Save Checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch + 1)

        # Final Save
        self._save_checkpoint("final")
        print("🎉 [Training Complete] Model saved.")

    def _save_checkpoint(self, suffix):
        path = os.path.join(self.model_save_dir, f"student_model_{suffix}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"   💾 Saved: {path}")

if __name__ == "__main__":
    # For quick testing
    import sys
    if len(sys.argv) > 1:
        session = sys.argv[1]
        trainer = Trainer(session)
        trainer.train()