# Project MUSE - model_arch.py
# The "Student" Architecture: ResNet-34 U-Net (High-Fidelity Desktop Mode)
# (C) 2025 MUSE Corp. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class DecoderBlock(nn.Module):
    """
    [U-Net Decoder Block]
    - Upsample -> Concat (Skip Connection) -> Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Skip Connection 채널과 입력 채널을 합침
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        # 1. Upsample (Bilinear)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # 2. Concat Skip Connection
        if skip is not None:
            # 해상도가 미세하게 안 맞을 경우를 대비한 Padding (Safe-guard)
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
            x = torch.cat([x, skip], dim=1)
            
        return self.conv(x)

class MuseStudentModel(nn.Module):
    def __init__(self, num_keypoints=17):
        """
        [MUSE Student Model - High Fidelity Edition]
        - Backbone: ResNet-34 (ImageNet Pretrained)
        - Architecture: U-Net Style (4-Stage Skip Connections)
        - Target Resolution: 960 x 544 (Half-HD, 16:9)
        - Target Hardware: RTX 3060 or higher
        
        Why ResNet34?
        - MobileNet보다 고주파 대역(머리카락, 주름) 보존력이 월등함.
        - TensorRT 변환 시 연산 최적화(Folding)가 매우 잘 됨.
        """
        super().__init__()
        
        # 1. Backbone (Encoder)
        weights = ResNet34_Weights.DEFAULT
        base_model = resnet34(weights=weights)
        
        # Extract Layers for Skip Connections
        # Input: (B, 3, H, W)
        self.enc0 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu
        ) # -> 1/2 scale (64ch)
        
        self.maxpool = base_model.maxpool # -> 1/4 scale
        
        self.enc1 = base_model.layer1 # -> 1/4 scale (64ch)
        self.enc2 = base_model.layer2 # -> 1/8 scale (128ch)
        self.enc3 = base_model.layer3 # -> 1/16 scale (256ch)
        self.enc4 = base_model.layer4 # -> 1/32 scale (512ch) - Bottleneck
        
        # 2. Decoder (U-Net Style)
        # Dec4: enc4(512) + enc3(256) -> 256
        self.dec4 = DecoderBlock(512, 256, 256)
        
        # Dec3: dec4(256) + enc2(128) -> 128
        self.dec3 = DecoderBlock(256, 128, 128)
        
        # Dec2: dec3(128) + enc1(64) -> 64
        self.dec2 = DecoderBlock(128, 64, 64)
        
        # Dec1: dec2(64) + enc0(64) -> 32 (Final Feature)
        # Note: enc0 is 1/2 scale. We need one more upsample to reach 1/1 scale?
        # Typically U-Net output is same size as input.
        # Here we target 1/2 or 1/4 scale for efficiency, but let's go for 1/2 scale (High Res).
        self.dec1 = DecoderBlock(64, 64, 32) 

        # Final Upsample Block to restore original resolution (1/2 -> 1/1)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 3. Heads
        # Head A: Segmentation (1 Channel)
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)
        
        # Head B: Pose Estimation (17 Channels)
        self.pose_head = nn.Conv2d(32, num_keypoints, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.enc0(x)      # (B, 64, H/2, W/2)
        x_pool = self.maxpool(x0)
        
        x1 = self.enc1(x_pool) # (B, 64, H/4, W/4)
        x2 = self.enc2(x1)     # (B, 128, H/8, W/8)
        x3 = self.enc3(x2)     # (B, 256, H/16, W/16)
        x4 = self.enc4(x3)     # (B, 512, H/32, W/32)
        
        # --- Decoder ---
        d4 = self.dec4(x4, x3) # -> 1/16
        d3 = self.dec3(d4, x2) # -> 1/8
        d2 = self.dec2(d3, x1) # -> 1/4
        d1 = self.dec1(d2, x0) # -> 1/2
        
        # Final Resolution Restoration (1/2 -> 1/1)
        # RTX 3060은 이정도 연산 충분히 감당 가능
        features = self.final_up(d1) # (B, 32, H, W)
        
        # --- Heads ---
        seg_logits = self.seg_head(features)   # (B, 1, H, W)
        pose_heatmaps = self.pose_head(features) # (B, 17, H, W)
        
        return seg_logits, pose_heatmaps