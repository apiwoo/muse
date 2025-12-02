# Project MUSE - train_student.py
# Multi-Profile Training Engine (High-Fidelity & Robust Overfitting & Distillation)
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import shutil
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# [Plan A] Robust Overfitting: Augmentation Library
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("⚠️ [Trainer] 'albumentations' 라이브러리가 없습니다. 데이터 증강이 비활성화됩니다.")
    print("   -> pip install albumentations")

# [MUSE Modules]
# Ensure path is added for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.ai.distillation.student.model_arch import MuseStudentModel

# [Plan A] Custom Dice Loss for Boundary Precision
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: sigmoid가 적용된 확률값 (B, 1, H, W)
        # target: 0 or 1 (B, 1, H, W)
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)
        
        loss = (1 - ((2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))
        return loss.mean()

# [Quality Upgrade] Knowledge Distillation Loss (Soft Labeling)
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=1.0):
        """
        :param alpha: Soft Label의 가중치 (0.0 ~ 1.0)
        :param temperature: Softening 강도 (높을수록 분포가 평탄해짐)
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.T = temperature
        self.bce_soft = nn.BCEWithLogitsLoss() # For Soft Targets (0~1)
        self.bce_hard = nn.BCEWithLogitsLoss() # For Hard Targets (0 or 1)
        self.mse = nn.MSELoss() # For Heatmaps

    def forward(self, student_seg, student_pose, soft_mask, hard_mask, soft_heatmap):
        """
        Segmentation: Soft Label(Blur) + Hard Label(Binary) 혼합 학습
        Pose: Confidence가 반영된 Soft Heatmap 학습 (MSE)
        """
        # 1. Segmentation Loss
        # Hard Loss: 정답을 맞추는 능력
        loss_seg_hard = self.bce_hard(student_seg, hard_mask)
        
        # Soft Loss: 경계면의 불확실성(Gradient)을 배우는 능력
        # Teacher의 Logits 대신 Smoothed Mask를 Soft Target으로 사용
        loss_seg_soft = self.bce_soft(student_seg, soft_mask)
        
        loss_seg_total = self.alpha * loss_seg_soft + (1.0 - self.alpha) * loss_seg_hard

        # 2. Pose Loss (Regression)
        # Heatmap은 이미 Confidence가 반영된 Soft Target이므로 MSE 사용
        loss_pose = self.mse(student_pose, soft_heatmap) * 10000.0 # Scale Up

        return loss_seg_total + loss_pose

class MuseDataset(Dataset):
    def __init__(self, profile_dir, input_size=(960, 544), heatmap_size=(960, 544), is_train=True):
        """
        [High-Fidelity Dataset with Augmentation & Soft Labels]
        - Input: 960x544 (Half-HD, 16:9)
        - Heatmap: 960x544 (1:1 Scale for ResNet-UNet)
        """
        self.img_dir = os.path.join(profile_dir, "images")
        self.mask_dir = os.path.join(profile_dir, "masks")
        self.label_dir = os.path.join(profile_dir, "labels")
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.is_train = is_train
        
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        print(f"   📂 [Dataset] 로드된 샘플 수: {len(self.img_files)}장 (Res: {input_size})")

        # [Plan A] Define Augmentation Pipeline
        # 강건한 개인화 모델을 위해 형태적 특징은 유지하되 픽셀값 변화를 줌
        if HAS_ALBUMENTATIONS and self.is_train:
            self.transform = A.Compose([
                # 1. 색상 변화 (옷 색깔이 미세하게 바뀔 때 대응)
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                
                # 2. 밝기/대비 (조명 변화 대응)
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                
                # 3. 미세한 기하학적 변형 (자세 틀어짐 대응)
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                
                # 4. 노이즈 (웹캠 화질 저하 대응)
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                
                # 5. 최종 리사이즈 (항상 마지막에)
                A.Resize(height=input_size[1], width=input_size[0]),
                
                # 6. 텐서 변환 및 정규화
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets={'mask': 'image', 'keypoints': 'keypoints'})
        else:
            # 검증용 혹은 라이브러리 없을 때: 단순 리사이즈만
            self.transform = A.Compose([
                A.Resize(height=input_size[1], width=input_size[0]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets={'mask': 'image', 'keypoints': 'keypoints'})

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{basename}.png")
        label_path = os.path.join(self.label_dir, f"{basename}.json")

        # 1. Image Load
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]

        # 2. Mask Load
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

        # 3. Keypoints Load
        # Albumentations requires format: list of [x, y]
        kpts_coord = []
        kpts_conf = []
        
        if os.path.exists(label_path):
             with open(label_path, 'r') as f:
                data = json.load(f)
                for kp in data['keypoints']:
                    kpts_coord.append([kp[0], kp[1]])
                    kpts_conf.append(kp[2])
        
        try:
            if HAS_ALBUMENTATIONS:
                transformed = self.transform(
                    image=img, 
                    mask=mask, 
                    keypoints=kpts_coord
                )
                img_tensor = transformed['image']
                
                # Mask Processing
                raw_mask = transformed['mask'] # (H, W) uint8
                
                # Hard Mask (Binary)
                mask_tensor = raw_mask.float().unsqueeze(0) / 255.0
                hard_mask_tensor = (mask_tensor > 0.5).float()
                
                # [Quality] Soft Mask (Label Smoothing)
                # 경계면을 부드럽게 만들어 Teacher의 'Soft Label' 효과를 냄
                # Gaussian Blur 적용 후 Tensor 변환
                soft_mask_cv = cv2.GaussianBlur(raw_mask.numpy(), (5, 5), 0)
                soft_mask_tensor = torch.from_numpy(soft_mask_cv).float().unsqueeze(0) / 255.0
                
                transformed_kpts = transformed['keypoints']
            else:
                # Fallback implementation (Manual Resize)
                img_resized = cv2.resize(img, self.input_size)
                mask_resized = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
                
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std
                
                mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0) / 255.0
                hard_mask_tensor = (mask_tensor > 0.5).float()
                soft_mask_tensor = mask_tensor # No blur fallback
                
                scale_x = self.input_size[0] / w_orig
                scale_y = self.input_size[1] / h_orig
                transformed_kpts = []
                for kp in kpts_coord:
                    transformed_kpts.append([kp[0] * scale_x, kp[1] * scale_y])

            # 4. Generate Confidence-Aware Heatmaps (High-Res)
            # Soft Labeling: Use confidence as peak value
            heatmaps = np.zeros((17, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
            
            num_kpts = min(len(transformed_kpts), len(kpts_conf))
            
            for i in range(num_kpts):
                x, y = transformed_kpts[i]
                conf = kpts_conf[i]
                
                # Check bounds
                if conf > 0.2 and 0 <= x < self.heatmap_size[0] and 0 <= y < self.heatmap_size[1]:
                    # [Quality] Peak를 1.0이 아닌 conf로 설정하여 '확신' 정도를 학습
                    self._add_gaussian(heatmaps[i], int(x), int(y), sigma=3.0, peak=conf)

            heatmap_tensor = torch.from_numpy(heatmaps)
            return img_tensor, soft_mask_tensor, hard_mask_tensor, heatmap_tensor

        except Exception as e:
            # Augmentation 실패 시 안전 리턴
            # print(f"⚠️ Augmentation Error at {basename}: {e}")
            empty_img = torch.zeros((3, self.input_size[1], self.input_size[0]))
            empty_mask = torch.zeros((1, self.input_size[1], self.input_size[0]))
            empty_hm = torch.zeros((17, self.heatmap_size[1], self.heatmap_size[0]))
            return empty_img, empty_mask, empty_mask, empty_hm


    def _add_gaussian(self, heatmap, x, y, sigma=3, peak=1.0):
        h, w = heatmap.shape
        tmp_size = sigma * 3
        mu_x = int(x + 0.5)
        mu_y = int(y + 0.5)
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        
        if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0: return heatmap
        
        size = 2 * tmp_size + 1
        x_vec = np.arange(0, size, 1, np.float32)
        y_vec = x_vec[:, np.newaxis]
        x0 = y0 = size // 2
        
        g = np.exp(- ((x_vec - x0) ** 2 + (y_vec - y0) ** 2) / (2 * sigma ** 2))
        
        # [Quality] Apply Confidence Peak
        g = g * peak
        
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
    def __init__(self, root_session="personal_data", epochs=50, batch_size=4):
        """
        [High-Fidelity Trainer]
        - batch_size: Default to 4 (Due to 960x544 resolution & ResNet34)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 [Trainer] Device: {self.device}")
        print(f"   - Target Resolution: 960x544 (High-Fidelity)")
        print(f"   - Batch Size: {batch_size}")
        
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.root_data_dir = os.path.join(self.root_dir, "recorded_data", root_session)
        self.model_save_dir = os.path.join(self.root_dir, "assets", "models", "personal")
        
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Check profiles
        if not os.path.exists(self.root_data_dir):
            print(f"❌ 데이터 폴더 없음: {self.root_data_dir}")
            sys.exit(1)
            
        self.profiles = [
            d for d in os.listdir(self.root_data_dir) 
            if os.path.isdir(os.path.join(self.root_data_dir, d))
        ]
        
        self.epochs = epochs
        self.batch_size = batch_size
        
        # [Plan A] Combined Loss Function
        # DiceLoss for boundary precision
        self.dice_loss = DiceLoss() 
        # [Quality] Distillation Loss for Soft Learning
        self.distill_loss = DistillationLoss(alpha=0.5, temperature=1.0)

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
            dataset = MuseDataset(profile_dir, is_train=True) # Uses new defaults 960x544
            if len(dataset) == 0:
                print(f"   ⚠️ 데이터가 없어 건너뜁니다.")
                return
            dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=4, 
                pin_memory=True
            )
        except Exception as e:
            print(f"   ❌ 데이터 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return

        # Model Init (ResNet-34 U-Net)
        model = MuseStudentModel(num_keypoints=17).to(self.device)
        
        # Smart Load (Fine-tune)
        if os.path.exists(model_path):
            print(f"   🔄 기존 모델 발견! 이어서 학습합니다: {model_name}")
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"   ⚠️ 모델 로드 실패 (구조 변경됨?): {e}")
                print("   -> 새로 학습합니다.")
        else:
            print(f"   ✨ 새로운 모델 학습 시작: {model_name}")

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Ep {epoch+1}/{self.epochs}", leave=False)
            
            for imgs, soft_masks, hard_masks, heatmaps in pbar:
                imgs = imgs.to(self.device)
                soft_masks = soft_masks.to(self.device)
                hard_masks = hard_masks.to(self.device)
                heatmaps = heatmaps.to(self.device)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    pred_seg, pred_pose = model(imgs)
                    
                    # [Quality] 1. Distillation Loss (Soft+Hard)
                    loss_distill = self.distill_loss(
                        pred_seg, pred_pose, 
                        soft_masks, hard_masks, 
                        heatmaps # heatmaps는 이미 Soft Target(Confidence 반영됨)
                    )
                    
                    # [Quality] 2. Dice Loss (Shape Consistency)
                    # Dice는 Binary Mask와 비교하는 것이 형태 학습에 유리
                    pred_seg_sigmoid = torch.sigmoid(pred_seg)
                    loss_dice = self.dice_loss(pred_seg_sigmoid, hard_masks)
                    
                    total_loss = loss_distill + loss_dice

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                loss_val = total_loss.item()
                running_loss += loss_val
                pbar.set_postfix({'loss': f"{loss_val:.4f}"})
            
            scheduler.step()
            
            # Simple epoch summary
            avg_loss = running_loss / len(dataloader)

        # Save
        if os.path.exists(model_path):
            backup_dir = os.path.join(self.model_save_dir, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copy2(model_path, os.path.join(backup_dir, f"backup_{model_name}"))
            
        torch.save(model.state_dict(), model_path)
        print(f"   🎉 [{profile}] 학습 완료! 저장됨: {model_name}")

if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "personal_data"
    # Allow command line override for batch size if needed
    batch_size = 4
    if len(sys.argv) > 2:
        try:
            batch_size = int(sys.argv[2])
        except: pass
        
    trainer = Trainer(session, epochs=50, batch_size=batch_size)
    trainer.train_all_profiles()