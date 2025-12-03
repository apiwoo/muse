# Project MUSE - train_student.py
# Multi-Profile Training Engine (SegFormer Edition - Full)
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import shutil
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob
import json

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

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, student_seg, student_pose, hard_mask, heatmaps):
        loss_seg = self.bce(student_seg, hard_mask)
        loss_pose = self.mse(student_pose, heatmaps) * 1000.0
        return loss_seg + loss_pose

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
            ], additional_targets={'mask': 'image', 'keypoints': 'keypoints'})
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

        kpts_coord = []
        kpts_conf = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                data = json.load(f)
                for kp in data['keypoints']:
                    kpts_coord.append([kp[0], kp[1]])
                    kpts_conf.append(kp[2])

        if HAS_ALBUMENTATIONS:
            transformed = self.transform(image=img, mask=mask, keypoints=kpts_coord)
            img_tensor = transformed['image']
            mask_tensor = transformed['mask'].float().unsqueeze(0) / 255.0
            mask_tensor = (mask_tensor > 0.5).float()
            transformed_kpts = transformed['keypoints']
        else:
            img_resized = cv2.resize(img, self.input_size)
            mask_resized = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
            img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0) / 255.0
            scale_x = self.input_size[0] / w_orig
            scale_y = self.input_size[1] / h_orig
            transformed_kpts = []
            for kp in kpts_coord:
                transformed_kpts.append([kp[0] * scale_x, kp[1] * scale_y])

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
    def __init__(self, root_session="personal_data", epochs=30, batch_size=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "recorded_data", root_session)
        self.model_save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "assets", "models", "personal")
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.profiles = [d for d in os.listdir(self.root_data_dir) if os.path.isdir(os.path.join(self.root_data_dir, d))]
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = DistillationLoss()
        self.dice_loss = DiceLoss()

    def train_all_profiles(self):
        total_profiles = len(self.profiles)
        for i, profile in enumerate(self.profiles):
            print(f"--- Profile ({i+1}/{total_profiles}): {profile} ---")
            self._train_single_profile(profile, profile_idx=i, total_profiles=total_profiles)

    def _train_single_profile(self, profile, profile_idx, total_profiles):
        print(f"\n🔥 Training SegFormer for [{profile}]...")
        dataset = MuseDataset(os.path.join(self.root_data_dir, profile))
        if len(dataset) == 0: return
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
        model = MuseStudentModel(num_keypoints=17, pretrained=True).to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            
            # [GUI Integration] Progress Calculation
            # 전체 진행률 = (현재 프로파일 완료율 + 현재 에폭 완료율) / 전체 프로파일 수
            # 하지만 간단하게: 현재 프로파일 내 에폭 진행률만 표시하거나, 전체 통합 표시
            # 여기서는 Studio가 단순 파싱하므로 Epoch 단위로 로그를 찍습니다.
            
            pbar = tqdm(dataloader, desc=f"Ep {epoch+1}/{self.epochs}", leave=False)
            
            for imgs, masks, heatmaps in pbar:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                heatmaps = heatmaps.to(self.device)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    pred_seg, pred_pose = model(imgs)
                    loss_dice = self.dice_loss(pred_seg, masks)
                    loss_total = self.loss_fn(pred_seg, pred_pose, masks, heatmaps) + loss_dice
                
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss_total.item()
                pbar.set_postfix({'loss': f"{loss_total.item():.4f}"})
            
            scheduler.step()
            
            # [GUI Log Format]
            # 전체 공정 중 현재 위치 계산
            # step_per_profile = 100 / total_profiles
            # current_base = step_per_profile * profile_idx
            # current_progress = current_base + (step_per_profile * (epoch + 1) / self.epochs)
            
            current_progress = int(((profile_idx * self.epochs) + (epoch + 1)) / (total_profiles * self.epochs) * 100)
            print(f"[PROGRESS] {current_progress}")
            print(f"   Epoch {epoch+1}/{self.epochs} - Loss: {running_loss/len(dataloader):.4f}")

        save_path = os.path.join(self.model_save_dir, f"student_{profile}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"   🎉 Model Saved: {save_path}")

if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "personal_data"
    Trainer(session).train_all_profiles()