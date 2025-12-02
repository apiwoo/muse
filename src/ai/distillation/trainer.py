# Project MUSE - train_student.py
# Multi-Profile Training Engine (Generates multiple models per profile)
# (C) 2025 MUSE Corp. All rights reserved.

import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import shutil
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# [MUSE Modules]
from src.ai.distillation.student.model_arch import MuseStudentModel

class MuseDataset(Dataset):
    def __init__(self, profile_dir, input_size=(512, 512), heatmap_size=(128, 128)):
        self.img_dir = os.path.join(profile_dir, "images")
        self.mask_dir = os.path.join(profile_dir, "masks")
        self.label_dir = os.path.join(profile_dir, "labels")
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        print(f"   📂 [Dataset] 로드된 샘플 수: {len(self.img_files)}장")

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

        keypoints = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                data = json.load(f)
                keypoints = np.array(data['keypoints'])
        
        img_resized = cv2.resize(img, self.input_size)
        mask_resized = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
        
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0) / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        heatmaps = np.zeros((17, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        if len(keypoints) > 0:
            scale_x = self.heatmap_size[0] / w_orig
            scale_y = self.heatmap_size[1] / h_orig
            for k_idx, (x, y, conf) in enumerate(keypoints):
                if conf > 0.4:
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
        if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0: return heatmap
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
    def __init__(self, root_session="personal_data", epochs=50, batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 [Trainer] Device: {self.device}")
        
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.root_data_dir = os.path.join(self.root_dir, "recorded_data", root_session)
        self.model_save_dir = os.path.join(self.root_dir, "assets", "models", "personal")
        
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # 프로파일 검색
        if not os.path.exists(self.root_data_dir):
            print(f"❌ 데이터 폴더 없음: {self.root_data_dir}")
            sys.exit(1)
            
        self.profiles = [
            d for d in os.listdir(self.root_data_dir) 
            if os.path.isdir(os.path.join(self.root_data_dir, d))
        ]
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def train_all_profiles(self):
        if not self.profiles:
            print("⚠️ 학습할 프로파일이 없습니다.")
            return

        print(f"🔥 총 {len(self.profiles)}개의 프로파일 학습을 시작합니다.")
        
        for profile in self.profiles:
            self._train_single_profile(profile)

    def _train_single_profile(self, profile):
        print(f"\n==================================================")
        print(f"   Training Profile: [{profile}]")
        print(f"==================================================")
        
        profile_dir = os.path.join(self.root_data_dir, profile)
        model_name = f"student_{profile}.pth"
        model_path = os.path.join(self.model_save_dir, model_name)
        
        # Dataset & Dataloader
        try:
            dataset = MuseDataset(profile_dir)
            if len(dataset) == 0:
                print(f"   ⚠️ 데이터가 없어 건너뜁니다.")
                return
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        except Exception as e:
            print(f"   ❌ 데이터 로드 실패: {e}")
            return

        # Model Init
        model = MuseStudentModel(num_keypoints=17).to(self.device)
        
        # Smart Load (Fine-tune)
        if os.path.exists(model_path):
            print(f"   🔄 기존 모델 발견! 이어서 학습합니다: {model_name}")
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            print(f"   ✨ 새로운 모델 학습 시작: {model_name}")

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Ep {epoch+1}/{self.epochs}", leave=False)
            
            for imgs, masks, heatmaps in pbar:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                heatmaps = heatmaps.to(self.device)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    pred_seg, pred_pose = model(imgs)
                    loss_seg = self.bce_loss(pred_seg, masks)
                    loss_pose = self.mse_loss(pred_pose, heatmaps) * 1000
                    total_loss = loss_seg + loss_pose

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += total_loss.item()
            
            scheduler.step()

        # Save
        # 기존 모델 백업
        if os.path.exists(model_path):
            backup_dir = os.path.join(self.model_save_dir, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copy2(model_path, os.path.join(backup_dir, f"backup_{model_name}"))
            
        torch.save(model.state_dict(), model_path)
        print(f"   🎉 [{profile}] 학습 완료! 저장됨: {model_name}")

if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "personal_data"
    trainer = Trainer(session, epochs=50, batch_size=8)
    trainer.train_all_profiles()