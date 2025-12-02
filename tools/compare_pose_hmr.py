# Project MUSE - compare_pose_hmr.py
# Purpose: Final Solution - Correct Keypoint Mapping (COCO vs OpenPose)
# (C) 2025 MUSE Corp.

import os
import sys
import shutil
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
from pathlib import Path
import traceback
import warnings 

warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

# 프로젝트 루트 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# [Dependencies Check]
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

if os.name == 'nt' and 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ.get('USERPROFILE', '.')

try:
    sys.modules["pyrender"] = MagicMock()
    sys.modules["pyrender.offscreen"] = MagicMock()
    os.environ["PYOPENGL_PLATFORM"] = "" 
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
except Exception as e:
    pass

# =========================================================
# 1. ViTPose Architecture
# =========================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=1280):
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
    def __init__(self, dim, num_heads=16, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
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

class ViTPose(nn.Module):
    def __init__(self, img_size=(256, 192), patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4., num_classes=17):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)) 
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.keypoint_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

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

# =========================================================
# 2. HMR Downloader
# =========================================================
def fix_hmr_missing_files():
    home = os.environ['HOME']
    cache_root = os.path.join(home, ".cache", "4DHumans")
    os.makedirs(cache_root, exist_ok=True)

    required_files = {
        "logs/train/multiruns/hmr2/0/model_config.yaml": "HMR2/logs/train/multiruns/hmr2/0/model_config.yaml",
        "logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt": "HMR2/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt",
        "data/smpl_mean_params.npz": "HMR2/data/smpl_mean_params.npz",
        "data/SMPL_to_J19.pkl": "HMR2/data/SMPL_to_J19.pkl",
        "data/smpl/SMPL_NEUTRAL.pkl": "HMR2/data/smpl/SMPL_NEUTRAL.pkl" 
    }
    
    for local_rel, remote_path in required_files.items():
        local_path = os.path.join(cache_root, local_rel)
        if not os.path.exists(local_path):
            try:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                hf_hub_download(repo_id="camenduru/4D-Humans", filename=remote_path, local_dir=cache_root, local_dir_use_symlinks=False)
                src = hf_hub_download(repo_id="camenduru/4D-Humans", filename=remote_path)
                shutil.copy(src, local_path)
            except Exception:
                pass

fix_hmr_missing_files()

try:
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
except ImportError:
    pass

# =========================================================
# 3. Helper Functions & Keypoint Definitions
# =========================================================
# [Correct] COCO 17 Format (For ViTPose Output)
COCO_NAMES = {
    0: "Nose", 1: "L-Eye", 2: "R-Eye", 3: "L-Ear", 4: "R-Ear",
    5: "L-Shldr", 6: "R-Shldr", 7: "L-Elbow", 8: "R-Elbow",
    9: "L-Wrist", 10: "R-Wrist", 11: "L-Hip", 12: "R-Hip",
    13: "L-Knee", 14: "R-Knee", 15: "L-Ankle", 16: "R-Ankle"
}

# [Correct] Mapping: HMR (OpenPose/SMPL) Index -> COCO Index
# HMR Index: 0:Nose, 1:Neck, 2:RShldr, 3:RElbow, 4:RWrist, 5:LShldr, 6:LElbow, 7:LWrist, 8:MidHip, 9:RHip...
# We map HMR index TO the corresponding COCO index to fill gaps
HMR_TO_COCO = {
    0: 0,   # Nose -> Nose
    16: 1,  # L-Eye -> L-Eye (HMR 16 is LEye)
    15: 2,  # R-Eye -> R-Eye (HMR 15 is REye)
    18: 3,  # L-Ear -> L-Ear
    17: 4,  # R-Ear -> R-Ear
    5: 5,   # L-Shldr -> L-Shldr
    2: 6,   # R-Shldr -> R-Shldr
    6: 7,   # L-Elbow -> L-Elbow
    3: 8,   # R-Elbow -> R-Elbow
    7: 9,   # L-Wrist -> L-Wrist
    4: 10,  # R-Wrist -> R-Wrist
    12: 11, # L-Hip -> L-Hip
    9: 12,  # R-Hip -> R-Hip
    13: 13, # L-Knee -> L-Knee
    10: 14, # R-Knee -> R-Knee
    14: 15, # L-Ankle -> L-Ankle
    11: 16  # R-Ankle -> R-Ankle
}

def get_bbox_from_kpts(kpts, img_w, img_h):
    if kpts is None or len(kpts) == 0: return None
    valid_pts = kpts[kpts[:, 2] > 0.1]
    if len(valid_pts) < 5: return None
    x1, y1 = np.min(valid_pts[:, 0]), np.min(valid_pts[:, 1])
    x2, y2 = np.max(valid_pts[:, 0]), np.max(valid_pts[:, 1])
    w = x2 - x1
    h = y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    size = max(w, h) * 1.2
    return cx, cy, size

def crop_image(img, cx, cy, size, target_size=256):
    h, w = img.shape[:2]
    x1 = int(cx - size / 2)
    y1 = int(cy - size / 2)
    x2 = int(cx + size / 2)
    y2 = int(cy + size / 2)
    
    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)
    
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(w, x2)
    crop_y2 = min(h, y2)
    
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        cropped = cv2.copyMakeBorder(cropped, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
        
    resized = cv2.resize(cropped, (target_size, target_size))
    return resized, (x1, y1, size)

# =========================================================
# 4. Hybrid Patching Logic (Corrected Mapping)
# =========================================================
def patch_hmr_onto_vitpose(kpts_hmr, kpts_vit):
    """
    HMR(OpenPose Format) -> ViTPose(COCO Format) 매핑 후 패치
    """
    if kpts_vit is None: return None
    
    # 1. HMR 데이터 유효성 체크
    if kpts_hmr is None: return kpts_vit
    
    # 2. 정렬 (Alignment) - Using COCO Indices for Anchor
    # ViTPose COCO: 5(LS), 6(RS), 11(LH), 12(RH)
    # HMR OpenPose: 5(LS), 2(RS), 12(LH), 9(RH)
    
    anchors_coco = [5, 6, 11, 12]
    anchors_hmr  = [5, 2, 12, 9] # Corresponding HMR indices
    
    shifts = []
    
    for c_idx, h_idx in zip(anchors_coco, anchors_hmr):
        if c_idx < len(kpts_vit) and kpts_vit[c_idx][2] > 0.4:
            if h_idx < len(kpts_hmr):
                diff = kpts_vit[c_idx][:2] - kpts_hmr[h_idx][:2]
                shifts.append(diff)
            
    if not shifts:
        return kpts_vit 
        
    avg_shift = np.mean(shifts, axis=0)
    
    # HMR 이동
    kpts_hmr_shifted = kpts_hmr.copy()
    kpts_hmr_shifted[:, :2] += avg_shift
    
    # 3. 패치 (Patch) - Mapping 적용
    kpts_final = kpts_vit.copy()
    
    for hmr_idx, coco_idx in HMR_TO_COCO.items():
        if hmr_idx >= len(kpts_hmr_shifted): continue
        
        # ViTPose 신뢰도가 낮으면(0.3 미만) HMR로 교체
        if kpts_vit[coco_idx][2] < 0.3:
            kpts_final[coco_idx][:2] = kpts_hmr_shifted[hmr_idx][:2]
            kpts_final[coco_idx][2] = 0.5 # Mark as HMR-patched
            
    return kpts_final

def run_comparison(image_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"🚀 Device: {device}")

    if not os.path.exists(image_path):
        print(f"❌ 파일 없음: {image_path}")
        return

    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None: return
    
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]

    # =================================================
    # [Step 1] ViTPose (2D)
    # =================================================
    print("🤖 [1/3] Running ViTPose (COCO Format)...")
    kpts_vit = None
    try:
        vit_model_path = os.path.join(BASE_DIR, "assets", "models", "tracking", "vitpose_huge_coco_256x192.pth")
        if os.path.exists(vit_model_path):
            vit_model = ViTPose(img_size=(256, 192), patch_size=16, embed_dim=1280, depth=32, num_heads=16, num_classes=17).to(device)
            state_dict = torch.load(vit_model_path, map_location=device, weights_only=False)
            if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            
            new_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('backbone.', '')
                if 'keypoint_head' in new_k:
                    new_k = new_k.replace('deconv_layers.0.', '0.').replace('deconv_layers.1.', '1.').replace('deconv_layers.3.', '3.').replace('deconv_layers.4.', '4.').replace('final_layer.', '6.')
                new_dict[new_k] = v
            vit_model.load_state_dict(new_dict, strict=False)
            vit_model.eval()
            
            crop_size = min(h_orig, w_orig)
            start_x = (w_orig - crop_size) // 2
            img_vit_in = img_rgb[:, start_x:start_x+crop_size] if w_orig > h_orig else img_rgb
            img_vit_resized = cv2.resize(img_vit_in, (192, 256))
            
            vit_tensor = torch.from_numpy(img_vit_resized.transpose(2, 0, 1)).float().to(device) / 255.0
            vit_tensor = (vit_tensor - torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)) / \
                         torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)
            
            with torch.no_grad():
                out_vit = vit_model(vit_tensor.unsqueeze(0))
            
            heatmaps = out_vit.cpu().numpy()[0]
            kpts_vit = []
            for i in range(17):
                hm = heatmaps[i]
                _, max_val, _, max_loc = cv2.minMaxLoc(hm)
                if max_val > 0.1:
                    x = max_loc[0] * (crop_size / 48.0) + (start_x if w_orig > h_orig else 0)
                    y = max_loc[1] * (crop_size / 64.0)
                    kpts_vit.append([x, y, max_val])
                else:
                    kpts_vit.append([0, 0, 0])
            kpts_vit = np.array(kpts_vit)
            
    except Exception as e:
        print(f"   ❌ ViTPose Error: {e}")
        traceback.print_exc()

    # =================================================
    # [Step 2] HMR (3D)
    # =================================================
    print("🤖 [2/3] Running HMR2 (OpenPose Format)...")
    kpts_hmr_raw = None
    
    try:
        # Guided Crop
        if kpts_vit is not None:
            valid_vit = kpts_vit[kpts_vit[:, 2] > 0.1]
            if len(valid_vit) > 0:
                cx, cy = np.mean(valid_vit[:, 0]), np.mean(valid_vit[:, 1])
                # HMR Crop
                crop_size = min(h_orig, w_orig) * 1.1 # Slightly larger
                img_hmr_in, crop_info = crop_image(img_rgb, cx, cy, crop_size)
                crop_x1, crop_y1, crop_s = crop_info
            else:
                raise ValueError("No ViTPose detections")
        else:
            raise ValueError("ViTPose failed")

        hmr_model, _ = load_hmr2(DEFAULT_CHECKPOINT)
        hmr_model = hmr_model.to(device)
        hmr_model.eval()

        img_tensor = torch.from_numpy(img_hmr_in.transpose(2, 0, 1)).float().to(device) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)
        img_tensor = (img_tensor - mean) / std
        
        with torch.no_grad():
            out = hmr_model({'img': img_tensor.unsqueeze(0)})
            pred_keypoints_3d = out['pred_keypoints_2d']

        raw_kpts = pred_keypoints_3d.cpu().numpy()[0]
        
        kpts_256 = (raw_kpts + 1) * 0.5 * 256.0
        scale = crop_s / 256.0
        kpts_hmr_raw = kpts_256 * scale
        kpts_hmr_raw[:, 0] += crop_x1
        kpts_hmr_raw[:, 1] += crop_y1
        
    except Exception as e:
        print(f"   ⚠️ HMR Warning: {e}")

    # =================================================
    # [Step 3] Patching
    # =================================================
    print("🔧 [3/3] Merging: Mapping HMR to COCO Format...")
    kpts_final = patch_hmr_onto_vitpose(kpts_hmr_raw, kpts_vit)

    # =================================================
    # Visualization
    # =================================================
    fig = plt.figure(figsize=(20, 10))
    
    # 1. ViTPose
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_rgb)
    ax1.set_title(f"ViTPose (COCO)\nLeft Shoulder={COCO_NAMES[5]}", fontsize=14)
    if kpts_vit is not None:
        for i, (x, y, conf) in enumerate(kpts_vit):
            if conf > 0.2:
                ax1.scatter(x, y, s=40, c='green', edgecolors='white', zorder=5)
                ax1.text(x, y, COCO_NAMES.get(i, str(i)), color='cyan', fontsize=9)

    # 2. HMR Raw
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(img_rgb)
    ax2.set_title("HMR Raw (OpenPose Format)", fontsize=14)
    if kpts_hmr_raw is not None:
        for i, pt in enumerate(kpts_hmr_raw):
            if i > 14: continue 
            x, y = pt[:2]
            ax2.scatter(x, y, s=40, c='gray', alpha=0.7)
            # HMR Index Display
            ax2.text(x, y, str(i), color='orange', fontsize=9)

    # 3. Hybrid
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(img_rgb)
    ax3.set_title("Final Hybrid (Correct Labels)", fontsize=14)
    
    out_info = []
    log_lines = ["=== Final COCO Pose Log ==="]
    
    if kpts_final is not None:
        for i, pt in enumerate(kpts_final):
            x, y = pt[:2]
            conf = pt[2]
            name = COCO_NAMES.get(i, f"Pt{i}")
            
            is_out = x < 0 or x > w_orig or y < 0 or y > h_orig
            
            color = 'green' # ViTPose
            if conf == 0.5: color = 'red' # HMR Patch
            
            if is_out:
                color = 'yellow'
                out_info.append(name)
                cx, cy = np.clip(x, 0, w_orig), np.clip(y, 0, h_orig)
                ax3.arrow(cx, cy, (x-cx)*0.2, (y-cy)*0.2, head_width=20, fc='yellow', ec='yellow', width=4)
                ax3.text(cx, cy, name, color='yellow', fontsize=10, fontweight='bold')
            else:
                if i in [0, 5, 6, 9, 10]: # Nose, Shoulders, Wrists
                    ax3.text(x, y, name, color='white', fontsize=9, fontweight='bold')

            ax3.scatter(x, y, s=50, c=color, edgecolors='black', zorder=10)
            log_lines.append(f"{i:<4d} {name:<10} {x:<6.0f} {y:<6.0f} {'OUT' if is_out else 'IN'}")

    plt.tight_layout()
    save_path = "hmr_hybrid_result.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    
    with open("pose_debug_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n==============================================")
    print("📊 [Result]")
    print(f"   ✅ 포맷 교정 완료 (HMR OpenPose -> ViTPose COCO)")
    print(f"   🖼️  결과 이미지: {os.path.abspath(save_path)}")
    print(f"   📍 확장된(화면 밖) 부위: {len(out_info)}개")
    if out_info: print(f"      ({', '.join(out_info)})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/compare_pose_hmr.py <image>")
    else:
        run_comparison(sys.argv[1])