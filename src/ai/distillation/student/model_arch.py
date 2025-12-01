# Project MUSE - model_arch.py
# The "Student" Architecture: MobileNetV3 Multi-Task Network
# (C) 2025 MUSE Corp. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class MuseStudentModel(nn.Module):
    def __init__(self, num_keypoints=17):
        """
        [MUSE Student Model]
        - Backbone: MobileNetV3-Large (Pretrained)
        - Head A (Segmentation): Background Removal (1 Channel)
        - Head B (Pose): Keypoint Heatmaps (17 Channels)
        """
        super().__init__()
        
        # 1. Backbone (Encoder)
        # Use ImageNet pretrained weights for faster convergence
        weights = MobileNet_V3_Large_Weights.DEFAULT
        base_model = mobilenet_v3_large(weights=weights)
        
        # Extract features from intermediate layers for skip connections (U-Net style)
        # MobileNetV3 layer indices depends on implementation, usually:
        # Low-level: features[3] (24ch, 1/4 scale)
        # High-level: features[16] (960ch, 1/32 scale)
        self.backbone = base_model.features
        
        # 2. Shared Decoder Layers (reduce channels)
        self.low_level_proj = nn.Conv2d(24, 48, kernel_size=1, bias=False)
        self.high_level_proj = nn.Conv2d(960, 256, kernel_size=1, bias=False)
        self.bn_high = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        # 3. Head A: Segmentation (Binary Mask)
        self.seg_head = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), # 1/4 -> Original
            nn.Conv2d(128, 1, kernel_size=1) # Output: 1 channel (Logits)
        )

        # 4. Head B: Pose Estimation (Heatmaps)
        self.pose_head = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False), # 1/8 -> 1/4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, kernel_size=1) # Output: 17 channels
        )

    def forward(self, x):
        input_size = x.shape[-2:] # (H, W)

        # Backbone Feature Extraction
        # MobileNetV3-Large specific feature extraction
        # We need to run through layers to get low and high level features
        low_level_feat = None
        high_level_feat = x
        
        for i, layer in enumerate(self.backbone):
            high_level_feat = layer(high_level_feat)
            if i == 3: # 1/4 scale feature
                low_level_feat = high_level_feat

        # --- Decoder Stage ---
        
        # 1. High-level feature processing (1/32 scale)
        x_high = self.high_level_proj(high_level_feat)
        x_high = self.bn_high(x_high)
        x_high = self.relu(x_high)
        
        # Upsample high-level to 1/4 scale to match low-level
        x_high_up = F.interpolate(x_high, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)
        
        # 2. Low-level feature processing (1/4 scale)
        x_low = self.low_level_proj(low_level_feat)
        
        # Concatenate (Skip Connection)
        x_cat = torch.cat([x_high_up, x_low], dim=1) # 256 + 48 = 304 channels

        # --- Heads ---
        
        # Head A: Segmentation (Full Resolution)
        seg_logits = self.seg_head(x_cat)
        
        # Head B: Pose (1/4 Resolution)
        pose_heatmaps = self.pose_head(x_cat)

        return seg_logits, pose_heatmaps