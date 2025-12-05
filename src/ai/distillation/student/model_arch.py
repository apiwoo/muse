# Project MUSE - model_arch.py
# The "Student" Architecture: SegFormer (MiT-B1)
# Replaces ResNet-34 U-Net for Global Context Awareness
# (C) 2025 MUSE Corp. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MuseStudentModel(nn.Module):
    def __init__(self, num_keypoints=17, pretrained=True):
        """
        [MUSE SegFormer Student]
        - Encoder: MiT-B1 (Mix Transformer)
        - Decoder: Lightweight MLP Decoder (Segmentation & Pose)
        - Resolution: 960 x 544
        """
        super().__init__()
        
        # 1. Encoder (MiT-B1)
        print("[STUDENT] [Student] Initializing SegFormer (MiT-B1)...")
        self.encoder = timm.create_model(
            'mit_b1', 
            pretrained=pretrained, 
            features_only=True
        )
        
        # Channel counts for MiT-B1: [64, 128, 320, 512]
        enc_channels = self.encoder.feature_info.channels()
        embedding_dim = 256

        # 2. MLP Decoder Layers
        self.linear_c4 = MLP(input_dim=enc_channels[3], embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=enc_channels[2], embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=enc_channels[1], embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=enc_channels[0], embed_dim=embedding_dim)

        self.dropout = nn.Dropout(0.1)
        
        self.linear_fuse = nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)

        # 3. Heads
        # Segmentation Head (1 Channel)
        self.seg_head = nn.Conv2d(embedding_dim, 1, kernel_size=1)
        
        # Pose Head (17 Channels)
        self.pose_head = nn.Conv2d(embedding_dim, num_keypoints, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, 544, 960)
        features = self.encoder(x)
        c1, c2, c3, c4 = features

        n, _, h, w = c4.shape 
        
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.shape[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.shape[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.shape[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.bn(_c)
        _c = self.relu(_c)
        _c = self.dropout(_c)

        seg_logits = self.seg_head(_c)
        pose_heatmaps = self.pose_head(_c)

        seg_logits = F.interpolate(seg_logits, scale_factor=4, mode='bilinear', align_corners=False)
        pose_heatmaps = F.interpolate(pose_heatmaps, scale_factor=4, mode='bilinear', align_corners=False)

        return seg_logits, pose_heatmaps

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x