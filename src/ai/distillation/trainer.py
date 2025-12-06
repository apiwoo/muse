# Project MUSE - trainer.py
# Multi-Profile Training Engine (Dual Mode & Single Profile Support)
# Updated: Detailed Loss Logging (BCE vs Dice)
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob
import json
import gc

# Ensure paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.ai.distillation.student.model_arch import MuseStudentModel

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))
        return loss.mean()

class MuseDataset(Dataset):
    def __init__(self, profile_dir, input_size=(960, 544)):
        self.img_dir = os.path.join(profile_dir, "images")
        self.mask_dir = os.path.join(profile_dir, "masks")
        self.label_dir = os.path.join(profile_dir, "labels")
        self.input_size = input_size
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))

        if HAS_ALBUMENTATIONS:
            self.transform = A.Compose([
                A.HueSaturationValue(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.Resize(height=input_size[1], width=input_size[0]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], 
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
            additional_targets={'mask': 'image', 'keypoints': 'keypoints'})
        else:
            self.transform = None

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{basename}.png")
        label_path = os.path.join(self.label_dir, f"{basename}.json")

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        
        # Robust Channel Handling
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h_orig, w_orig = img.shape[:2]
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        else:
            mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

        kpts_coord = []
        kpts_conf = []
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    data = json.load(f)
                    for kp in data['keypoints']:
                        kpts_coord.append([kp[0], kp[1]])
                        kpts_conf.append(kp[2])
            except Exception: pass
        
        if len(kpts_coord) == 0:
            kpts_coord = [[0, 0]] * 17
            kpts_conf = [0.0] * 17

        if HAS_ALBUMENTATIONS:
            try:
                transformed = self.transform(image=img, mask=mask, keypoints=kpts_coord)
                img_tensor = transformed['image']
                mask_tensor = transformed['mask'].float().unsqueeze(0) / 255.0
                mask_tensor = (mask_tensor > 0.5).float()
                transformed_kpts = transformed['keypoints']
            except Exception:
                # Fallback
                img_resized = cv2.resize(img, self.input_size)
                mask_resized = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
                img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std
                mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0) / 255.0
                scale_x = self.input_size[0] / w_orig
                scale_y = self.input_size[1] / h_orig
                transformed_kpts = [[kp[0] * scale_x, kp[1] * scale_y] for kp in kpts_coord]
        else:
            # Basic Fallback
            img_resized = cv2.resize(img, self.input_size)
            mask_resized = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
            img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float() / 255.0
            img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0) / 255.0
            scale_x = self.input_size[0] / w_orig
            scale_y = self.input_size[1] / h_orig
            transformed_kpts = [[kp[0] * scale_x, kp[1] * scale_y] for kp in kpts_coord]

        heatmaps = np.zeros((17, self.input_size[1], self.input_size[0]), dtype=np.float32)
        for i, (x, y) in enumerate(transformed_kpts):
            if i < len(kpts_conf) and kpts_conf[i] > 0.2:
                self._add_gaussian(heatmaps[i], int(x), int(y))
        
        heatmap_tensor = torch.from_numpy(heatmaps)
        return img_tensor, mask_tensor, heatmap_tensor

    def _add_gaussian(self, heatmap, x, y, sigma=3):
        h, w = heatmap.shape
        tmp_size = sigma * 3
        mu_x, mu_y = int(x + 0.5), int(y + 0.5)
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0: return heatmap
        size = 2 * tmp_size + 1
        x_vec = np.arange(0, size, 1, np.float32)
        y_vec = x_vec[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x_vec - x0) ** 2 + (y_vec - y0) ** 2) / (2 * sigma ** 2))
        g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
        img_x = max(0, ul[0]), min(br[0], w)
        img_y = max(0, ul[1]), min(br[1], h)
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]], g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
        return heatmap

class Trainer:
    def __init__(self, root_session="personal_data", task="seg", target_profile=None, epochs=50, batch_size=4):
        """
        Args:
            task (str): 'seg' or 'pose' (Determines model architecture and loss)
            target_profile (str): If set, only trains this profile.
        """
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.root_data_dir = os.path.join(base_path, "recorded_data", root_session)
        self.model_save_dir = os.path.join(base_path, "assets", "models", "personal")
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        self.profiles = [d for d in os.listdir(self.root_data_dir) if os.path.isdir(os.path.join(self.root_data_dir, d))]
        
        # Filter for target profile
        if target_profile:
            if target_profile in self.profiles:
                print(f"[TRAINER] Target Lock: Only training '{target_profile}'")
                self.profiles = [target_profile]
            else:
                print(f"[WARNING] Target '{target_profile}' not found in {self.root_data_dir}. Falling back to all.")

        self.epochs = epochs
        self.batch_size = batch_size
        
        # Loss Functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.dice_loss = DiceLoss()

        print(f"[TRAINER] Initialized for TASK: {self.task.upper()}")

    def train_all_profiles(self):
        total = len(self.profiles)
        for i, profile in enumerate(self.profiles):
            print(f"--- Profile ({i+1}/{total}): {profile} [{self.task.upper()}] ---")
            self._train_single_profile(profile, i, total)

    def _train_single_profile(self, profile, profile_idx, total_profiles):
        dataset = MuseDataset(os.path.join(self.root_data_dir, profile))
        if len(dataset) == 0: return
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        # Init Model based on Task
        model = MuseStudentModel(num_keypoints=17, pretrained=True, mode=self.task).to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        try:
            scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
        
        stop_flag = os.path.join(self.root_data_dir, "stop_training.flag")
        if os.path.exists(stop_flag): os.remove(stop_flag)

        for epoch in range(self.epochs):
            if os.path.exists(stop_flag):
                print(f"\n[STOP] User abort requested.")
                break

            model.train()
            run_loss = 0.0
            run_bce = 0.0
            run_dice = 0.0
            
            pbar = tqdm(dataloader, desc=f"Ep {epoch+1}/{self.epochs}", leave=False)
            
            for imgs, masks, heatmaps in pbar:
                imgs = imgs.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = model(imgs)
                    
                    if self.task == 'seg':
                        masks = masks.to(self.device, non_blocking=True)
                        l_bce = self.bce_loss(output, masks)
                        l_dice = self.dice_loss(output, masks)
                        loss = l_bce + (l_dice * 5.0)
                        
                        run_bce += l_bce.item()
                        run_dice += l_dice.item()
                        
                        # [Modified] Detailed Log for User
                        pbar.set_postfix_str(f"T:{loss.item():.4f} B:{l_bce.item():.4f} D:{l_dice.item():.4f}")
                        
                    elif self.task == 'pose':
                        heatmaps = heatmaps.to(self.device, non_blocking=True)
                        l_mse = self.mse_loss(output, heatmaps) * 3000.0 
                        loss = l_mse
                        pbar.set_postfix_str(f"Pose Loss: {loss.item():.4f}")
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                run_loss += loss.item()
            
            scheduler.step()
            gc.collect()
            
            current_progress = int(((profile_idx * self.epochs) + (epoch + 1)) / (total_profiles * self.epochs) * 100)
            print(f"[PROGRESS] {current_progress}")
            
            avg_loss = run_loss / len(dataloader)
            
            if self.task == 'seg':
                avg_bce = run_bce / len(dataloader)
                avg_dice = run_dice / len(dataloader)
                print(f"   Epoch {epoch+1}/{self.epochs} - Avg: {avg_loss:.4f} (BCE: {avg_bce:.4f}, Dice: {avg_dice:.4f})")
            else:
                print(f"   Epoch {epoch+1}/{self.epochs} - Avg Loss: {avg_loss:.4f}")

        # Save with suffix
        save_name = f"student_{self.task}_{profile}.pth"
        save_path = os.path.join(self.model_save_dir, save_name)
        torch.save(model.state_dict(), save_path)
        print(f"   [DONE] Saved: {save_name}")
        
        del model
        del optimizer
        del scaler
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    pass