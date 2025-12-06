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
import gc # [Fix] 메모리 관리를 위한 모듈 추가

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
        # [Log Fix] Pose Loss 스케일링 유지
        loss_pose = self.mse(student_pose, heatmaps) * 1000.0
        
        # [Log Fix] 개별 Loss 추적을 위해 튜플 반환 (Total, Seg, Pose)
        return loss_seg + loss_pose, loss_seg, loss_pose

class MuseDataset(Dataset):
    def __init__(self, profile_dir, input_size=(960, 544)):
        self.img_dir = os.path.join(profile_dir, "images")
        self.mask_dir = os.path.join(profile_dir, "masks")
        self.label_dir = os.path.join(profile_dir, "labels")
        self.input_size = input_size
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))

        if HAS_ALBUMENTATIONS:
            # [FIX] keypoint_params 추가: format='xy'로 설정하여 (x, y) 좌표만 사용함을 명시
            # remove_invisible=False: 이미지를 벗어난 키포인트도 유지 (인덱스 밀림 방지)
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

        # [FIX] 강력한 이미지 로드 및 채널 정규화 (Robust Loading)
        img = cv2.imread(img_path)
        
        # 1. 파일 로드 실패 시 더미 이미지 생성
        if img is None:
            img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        
        # 2. 채널 확인 및 RGB 변환 (Albumentations 오류 방지)
        if len(img.shape) == 2:
            # Grayscale (H, W) -> RGB (H, W, 3)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                # BGR (H, W, 3) -> RGB (H, W, 3)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                # BGRA (H, W, 4) -> RGB (H, W, 3) - 투명도 채널 제거
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                # 이상한 채널 수 -> 강제 Grayscale 후 RGB 변환 (안전장치)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            # 예상치 못한 차원 -> 강제 리셋
            img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)

        h_orig, w_orig = img.shape[:2]
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: # 마스크 로드 실패 시
                mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
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
            except Exception: # JSON 파싱 에러 등
                pass
        
        # 키포인트가 비어있으면 더미 데이터 채움 (17개)
        if len(kpts_coord) == 0:
            kpts_coord = [[0, 0]] * 17
            kpts_conf = [0.0] * 17

        if HAS_ALBUMENTATIONS:
            # Albumentations 적용
            try:
                transformed = self.transform(image=img, mask=mask, keypoints=kpts_coord)
                img_tensor = transformed['image']
                mask_tensor = transformed['mask'].float().unsqueeze(0) / 255.0
                mask_tensor = (mask_tensor > 0.5).float()
                transformed_kpts = transformed['keypoints']
            except Exception as e:
                # 변환 실패 시 원본 리사이즈로 폴백 (데이터 오염 방지)
                # print(f"[WARNING] Augmentation failed for {basename}: {e}") # 로그 과다 방지
                img_resized = cv2.resize(img, self.input_size)
                mask_resized = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
                img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float() / 255.0
                # Normalize manually
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std
                
                mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0) / 255.0
                scale_x = self.input_size[0] / w_orig
                scale_y = self.input_size[1] / h_orig
                transformed_kpts = [[kp[0] * scale_x, kp[1] * scale_y] for kp in kpts_coord]

        else:
            # Fallback (OpenCV Resize)
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
        print(f"\n[FIRE] Training SegFormer for [{profile}]...")
        dataset = MuseDataset(os.path.join(self.root_data_dir, profile))
        if len(dataset) == 0: return
        
        # [Optimization] pin_memory=True 추가 (CPU->GPU 전송 속도 향상)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        model = MuseStudentModel(num_keypoints=17, pretrained=True).to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        # [Fix] Deprecation Warning Fix
        try:
            scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
        
        # [New] Stop Flag Check Path
        stop_flag = os.path.join(self.root_data_dir, "stop_training.flag")
        # 시작 전 이전 플래그 삭제
        if os.path.exists(stop_flag): os.remove(stop_flag)

        for epoch in range(self.epochs):
            # [New] Early Stop Check
            if os.path.exists(stop_flag):
                print(f"\n[STOP] 사용자 중단 요청 감지. 학습을 조기 종료하고 모델을 저장합니다.")
                break

            model.train()
            
            # [Log Fix] 개별 Loss 추적 변수
            run_total = 0.0
            run_seg = 0.0
            run_dice = 0.0
            run_pose = 0.0
            
            pbar = tqdm(dataloader, desc=f"Ep {epoch+1}/{self.epochs}", leave=False)
            
            for imgs, masks, heatmaps in pbar:
                # [Optimization] non_blocking=True (비동기 전송)
                imgs = imgs.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                heatmaps = heatmaps.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    pred_seg, pred_pose = model(imgs)
                    
                    # [Log Fix] Loss 분리 계산 및 수집
                    # loss_fn 반환값: (distill_total, seg_bce, pose_mse)
                    distill_loss, l_seg, l_pose = self.loss_fn(pred_seg, pred_pose, masks, heatmaps)
                    
                    l_dice = self.dice_loss(pred_seg, masks)
                    
                    # 최종 Total Loss
                    loss_total = distill_loss + l_dice
                
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # [Log Fix] 누적 및 로그 업데이트
                run_total += loss_total.item()
                run_seg += l_seg.item()
                run_dice += l_dice.item()
                run_pose += l_pose.item()
                
                # 상세 로그 포맷: Total | Seg(BCE) | Dice | Pose
                pbar.set_postfix_str(
                    f"T:{loss_total.item():.4f} | S:{l_seg.item():.4f} | D:{l_dice.item():.4f} | P:{l_pose.item():.4f}"
                )
            
            scheduler.step()
            
            # [Fix] Epoch마다 PC 메모리 강제 정리 (누수 방지 핵심)
            gc.collect()
            
            current_progress = int(((profile_idx * self.epochs) + (epoch + 1)) / (total_profiles * self.epochs) * 100)
            print(f"[PROGRESS] {current_progress}")
            
            # Epoch 평균 로그 출력
            avg_len = len(dataloader)
            print(f"   Epoch {epoch+1}/{self.epochs} - "
                  f"Avg Loss: {run_total/avg_len:.4f} "
                  f"(Seg: {run_seg/avg_len:.4f}, Dice: {run_dice/avg_len:.4f}, Pose: {run_pose/avg_len:.4f})")

        save_path = os.path.join(self.model_save_dir, f"student_{profile}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"   [DONE] Model Saved: {save_path}")
        
        # [Fix] 프로파일 학습 종료 후 자원 명시적 해제
        del model
        del optimizer
        del scaler
        del dataloader
        del dataset
        
        gc.collect() # PC RAM 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # GPU VRAM 정리 (파편화 제거)

if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "personal_data"
    Trainer(session).train_all_profiles()