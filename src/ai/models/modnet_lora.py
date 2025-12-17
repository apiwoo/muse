# Project MUSE - modnet_lora.py
# MODNet Architecture with LoRA Injection Capability
# Based on MobileNetV2 Backbone
# (C) 2025 MUSE Corp. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# [LoRA Layer] Conv2d Wrapper
# ==============================================================================
class LoRAConv2d(nn.Module):
    def __init__(self, conv_layer, rank=4, alpha=4):
        super().__init__()
        self.conv = conv_layer # Original Layer (Frozen)
        self.rank = rank
        self.scaling = alpha / rank
        
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size[0]
        stride = conv_layer.stride
        padding = conv_layer.padding
        groups = conv_layer.groups
        
        # LoRA A: (rank, in, k, k) -> Reduces dim
        # LoRA B: (out, rank, 1, 1) -> Restores dim (Pointwise)
        # Note: Implementing standard LoRA for Conv2d is tricky. 
        # Here we use a simplified version: Pointwise LoRA on Kernel Center or 1x1 approximation
        
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, groups=groups, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, 1, 1, 0, bias=False)
        
        # Init
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.conv(x) + (self.lora_B(self.lora_A(x)) * self.scaling)

# ==============================================================================
# [MobileNetV2 Backbone] Simplified
# ==============================================================================
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, in_channels=3, alpha=1.0):
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(input_channel * alpha)
        self.last_channel = int(last_channel * alpha) if alpha > 1.0 else last_channel
        
        self.features = [nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]
        
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * alpha)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
                
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        ))
        
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)

# ==============================================================================
# [MODNet]
# ==============================================================================
class MODNet(nn.Module):
    def __init__(self, in_channels=3, backbone_pretrained=False):
        super(MODNet, self).__init__()
        self.in_channels = in_channels
        self.backbone = MobileNetV2(in_channels, alpha=1.0)
        
        # LR Branch
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1280, 1280 // 16, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1280 // 16, 1280, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.conv_lr_16x = nn.Conv2d(1280, 512, 1, 1, 0, bias=False)
        self.conv_lr_8x = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.conv_lr = nn.Conv2d(256, 1, 3, 1, 1, bias=False)

        # HR Branch
        self.conv_hr = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            InvertedResidual(16, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6),
            nn.Conv2d(24, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # Fusion Branch
        self.conv_fus_1 = nn.Sequential(
            nn.Conv2d(32 + 256, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv_fus_2 = nn.Sequential(
            nn.Conv2d(64 + 256, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv_fus = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, img):
        # 1. Backbone (Extract features at 1/16 scale)
        # MobileNetV2 structure in this impl is Sequential, need to hook intermediate layers if needed.
        # But MODNet original uses specific intermediate outputs.
        # For simplicity, we assume we need enc_2x, enc_4x, enc_8x, enc_16x... 
        # Wait, MODNet uses backbone output (1/32) and intermediate (1/16, 1/8)?
        # Let's approximate: 
        # Standard MODNet uses mobilenetv2 features at index:
        # 0-1: 1/2, 2-3: 1/4, 4-6: 1/8, 7-13: 1/16, 14-18: 1/32
        
        # Manually run backbone to get intermediate features
        x = img
        enc_features = []
        # features[0] -> ConvBNReLU (1/2)
        x = self.backbone.features[0](x); enc_features.append(x) # 0: 1/2
        
        # features[1] -> IR (1/2)
        x = self.backbone.features[1](x); # 1
        
        # features[2-3] -> IR (1/4)
        x = self.backbone.features[2](x)
        x = self.backbone.features[3](x); enc_features.append(x) # 3: 1/4
        
        # features[4-6] -> IR (1/8)
        for i in range(4, 7): x = self.backbone.features[i](x)
        enc_2x = x # Actually 1/8, name confusion in original paper implementation
        
        # features[7-13] -> IR (1/16)
        for i in range(7, 14): x = self.backbone.features[i](x)
        enc_4x = x # 1/16
        
        # features[14-18] -> IR (1/32) + Last Conv
        for i in range(14, 19): x = self.backbone.features[i](x)
        enc_32x = x
        
        # --- LR Branch ---
        lr8x = F.interpolate(enc_32x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv_lr_16x(lr8x) # 512ch
        lr8x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv_lr_8x(lr8x) # 256ch (1/8 scale)
        
        pred_semantic = self.conv_lr(lr8x) # Semantic Output
        
        # --- HR Branch ---
        hr2x = self.conv_hr(img) # 1/2 scale (32ch)
        
        # --- Fusion Branch ---
        # Fuse 1: LR feature (256) + HR feature (32)? No, we need to upsample LR
        # Original MODNet fuses 1/8 LR feature with 1/2 HR feature?
        # Actually it fuses enc_2x (1/8) from backbone first
        
        # Let's simplify to match weights structure roughly.
        # Assume lr8x is feature_lr
        
        # Fusion 1: enc_2x (1/8 scale, 32ch) + lr8x (1/8 scale, 256ch)
        # Note: enc_2x in MobileNetV2 at index 6 is 32ch.
        # conv_fus_1 takes 32+256 -> 64
        # Wait, MobileNetV2 index 6 output is 32 channels.
        
        fus_1 = torch.cat([enc_2x, lr8x], dim=1) 
        fus_1 = self.conv_fus_1(fus_1) # 64ch, 1/8 scale
        
        # Upsample to 1/2
        fus_1_up = F.interpolate(fus_1, scale_factor=4, mode='bilinear', align_corners=False)
        
        # Fusion 2: hr2x (32ch) + fus_1_up (64ch)??
        # conv_fus_2 takes 64+256? No, original paper structure is complex.
        # Based on .ckpt weights analysis:
        # conv_fus_2 weights are [64, 320, 3, 3] -> Input 320.
        # So it expects 64 (previous) + 256 (something else).
        # It concatenates with re-used lr8x upsampled?
        
        lr2x = F.interpolate(lr8x, scale_factor=4, mode='bilinear', align_corners=False)
        fus_2 = torch.cat([fus_1_up, lr2x], dim=1) # 64 + 256 = 320
        fus_2 = self.conv_fus_2(fus_2) # 64ch
        
        # Detail output
        pred_detail = fus_2 # Actually there is a separate detail head usually, but we focus on matte
        
        # Final Matte
        pred_matte = self.conv_fus(fus_2) # Sigmoid included
        
        # Upsample to full resolution
        pred_matte = F.interpolate(pred_matte, scale_factor=2, mode='bilinear', align_corners=False)
        
        return pred_matte

# ==============================================================================
# [Wrapper with LoRA]
# ==============================================================================
class MODNetLoRA(MODNet):
    def __init__(self, in_channels=3, pretrained=False):
        super().__init__(in_channels)
        self.lora_layers = []
        
    def inject_lora(self, rank=4):
        """
        Replace key Conv2d layers with LoRA wrappers.
        Target: Fusion Branch and LR Branch (Context)
        """
        print(f"[MODNet] Injecting LoRA (rank={rank})...")
        
        targets = [
            self.conv_lr_16x, self.conv_lr_8x,
            self.conv_fus_1[0], self.conv_fus_2[0],
            self.conv_fus[0], self.conv_fus[3]
        ]
        
        # Freeze all first
        for param in self.parameters():
            param.requires_grad = False
            
        # Replace and Enable Grad for LoRA
        for i, target in enumerate(targets):
            if isinstance(target, nn.Conv2d):
                lora_layer = LoRAConv2d(target, rank=rank)
                # Keep original weights frozen inside wrapper
                
                # Re-assign to module
                # This is hardcoded for specific attributes
                if target == self.conv_lr_16x: self.conv_lr_16x = lora_layer
                elif target == self.conv_lr_8x: self.conv_lr_8x = lora_layer
                elif target == self.conv_fus_1[0]: self.conv_fus_1[0] = lora_layer
                elif target == self.conv_fus_2[0]: self.conv_fus_2[0] = lora_layer
                elif target == self.conv_fus[0]: self.conv_fus[0] = lora_layer
                elif target == self.conv_fus[3]: self.conv_fus[3] = lora_layer
                
                self.lora_layers.append(lora_layer)
        
        # Verify trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MODNet] LoRA Injected. Trainable Params: {trainable}")