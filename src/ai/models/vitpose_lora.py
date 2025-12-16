# Project MUSE - vitpose_lora.py
# ViTPose Backbone with Low-Rank Adaptation (LoRA) Injection
# (C) 2025 MUSE Corp. All rights reserved.

import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    LoRA Layer: Injects trainable rank-decomposition matrices into Linear layers.
    W' = W + (B @ A) * scaling
    """
    def __init__(self, linear_layer, rank=8, alpha=16):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Freeze original weights
        linear_layer.weight.requires_grad = False
        if linear_layer.bias is not None:
            linear_layer.bias.requires_grad = False
        
        self.linear = linear_layer
        
        # LoRA weights (Trainable)
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original output
        out = self.linear(x)
        # LoRA output: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return out + lora_out

# --- ViTPose Architecture (Matched with Base Weights) ---

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTPoseLoRA(nn.Module):
    def __init__(self, img_size=(256, 192), patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., num_classes=17):
        super().__init__()
        # Base Model Config (ViT-Base)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)) 
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Simple Decoder Head
        self.keypoint_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

    def inject_lora(self, rank=8):
        """
        Replace Attention qkv linear layers with LoRA Layers.
        """
        print(f"[LoRA] Injecting LoRA layers (Rank={rank})...")
        for i, blk in enumerate(self.blocks):
            # Target: Attention qkv
            original_qkv = blk.attn.qkv
            blk.attn.qkv = LoRALayer(original_qkv, rank=rank)
            
        # Freeze everything except LoRA and Head
        for name, param in self.named_parameters():
            if 'lora_' in name or 'keypoint_head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) 
        pos_embed = self.pos_embed[:, 1:, :] 
        x = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        H, W = self.patch_embed.grid_size
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.keypoint_head(x)
        return x