# Project MUSE - model_arch.py
# The "Student" Architecture: SegFormer (MiT-B1)
# Replaces ResNet-34 U-Net for Global Context Awareness
# (C) 2025 MUSE Corp. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# [Mix Transformer (MiT)] Implementation
# timm 라이브러리 버전 이슈로 인해 직접 구현체를 포함시킵니다.
# ==============================================================================

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        
        # DropPath implementation (Identity if unavailable)
        self.drop_path = nn.Identity() # Simplification for portability
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # Patch Embeddings
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Transformer Encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x_out = self.norm1(x)
        x_out = x_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_out)

        # Stage 2
        x, H, W = self.patch_embed2(x_out)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x_out = self.norm2(x)
        x_out = x_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_out)

        # Stage 3
        x, H, W = self.patch_embed3(x_out)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x_out = self.norm3(x)
        x_out = x_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_out)

        # Stage 4
        x, H, W = self.patch_embed4(x_out)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x_out = self.norm4(x)
        x_out = x_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_out)

        return outs

# ==============================================================================
# [MUSE Student Model] Wrapper
# ==============================================================================

class MuseStudentModel(nn.Module):
    def __init__(self, num_keypoints=17, pretrained=True):
        """
        [MUSE SegFormer Student]
        - Encoder: MiT-B1 (Mix Transformer) [Direct Implementation]
        - Decoder: Lightweight MLP Decoder (Segmentation & Pose)
        - Resolution: 960 x 544
        """
        super().__init__()
        
        # 1. Encoder (MiT-B1)
        print("[STUDENT] [Student] Initializing SegFormer (MiT-B1) - Custom Impl...")
        # MiT-B1 Configuration
        self.encoder = MixVisionTransformer(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], 
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True, norm_layer=nn.LayerNorm, 
            depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1]
        )
        
        if pretrained:
            print("   [INFO] Pretrained weights are disabled for Custom MiT to avoid URL errors.")
            # For production, we should load state_dict from a local file if available.
        
        # Channel counts for MiT-B1: [64, 128, 320, 512]
        enc_channels = [64, 128, 320, 512]
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
        # MiT Encoder forward
        features = self.encoder.forward_features(x)
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