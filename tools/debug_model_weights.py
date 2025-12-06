# Project MUSE - debug_model_weights.py
# (C) 2025 MUSE Corp.
# Purpose: 학습된 .pth 모델의 가중치 상태 및 추론 출력값 정밀 진단
# "과연 모델이 학습이 된 건가, 아니면 변환이 잘못된 건가?"

import torch
import cv2
import os
import sys
import numpy as np
import glob

# Add path to src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ai.distillation.student.model_arch import MuseStudentModel

def debug_weights(profile_name):
    # Paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, "assets", "models", "personal", f"student_{profile_name}.pth")
    data_dir = os.path.join(root_dir, "recorded_data", "personal_data", profile_name, "images")
    
    print("========================================================")
    print(f"   MUSE Model Weight Inspector")
    print(f"   Target: {os.path.basename(model_path)}")
    print("========================================================")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    # 1. Model Load
    print("[1] Loading Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 학습 코드와 동일한 아키텍처 로드
        model = MuseStudentModel(num_keypoints=17, pretrained=False).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("   ✅ Weights loaded successfully.")
    except Exception as e:
        print(f"   ❌ Failed to load weights: {e}")
        return

    # 2. Image Load
    print("[2] Loading Sample Image...")
    img_files = glob.glob(os.path.join(data_dir, "*.jpg"))
    if not img_files:
        print("   ❌ No images found in profile data.")
        return
    
    # 첫 번째 이미지 사용
    img_path = img_files[0]
    print(f"   📸 Testing on: {os.path.basename(img_path)}")
    
    img = cv2.imread(img_path)
    h_orig, w_orig = img.shape[:2]
    
    # 3. Preprocessing (Inference Logic)
    input_size = (960, 544)
    img_resized = cv2.resize(img, input_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Normalize (ImageNet Mean/Std)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img_tensor = (img_tensor - mean) / std

    # 4. Forward Pass
    print("[3] Running Inference...")
    with torch.no_grad():
        pred_seg, pred_pose = model(img_tensor)
    
    # 5. Analysis
    print("\n================ [ DIAGNOSIS REPORT ] ================")
    
    # --- Segmentation ---
    seg_prob = torch.sigmoid(pred_seg).squeeze().cpu().numpy()
    seg_conf = seg_prob.max()
    print(f"🧩 Segmentation Output:")
    print(f"   - Max Probability: {seg_conf:.4f} (Should be near 1.0)")
    print(f"   - Min Probability: {seg_prob.min():.4f}")
    
    if seg_conf < 0.5:
        print("   ⚠️ [WARNING] Segmentation output is too weak!")
    
    # --- Pose (Heatmaps) ---
    pose_map = pred_pose.squeeze().cpu().numpy() # (17, H, W)
    pose_max = pose_map.max()
    pose_min = pose_map.min()
    
    print(f"\n🦴 Pose Heatmaps Output (Raw Values):")
    print(f"   - Max Value: {pose_max:.6f}")
    print(f"   - Min Value: {pose_min:.6f}")
    
    # [Critical Check]
    if pose_max < 0.05:
        print("   🚨 [CRITICAL FAIL] Max confidence is below 0.05.")
        print("      -> The model failed to learn keypoints.")
        print("      -> Possible Cause: Loss didn't converge, or learning rate too low.")
    elif pose_max < 0.3:
        print("   ⚠️ [WARNING] Confidence is low (0.05 ~ 0.3).")
        print("      -> It might work but will be unstable.")
    else:
        print("   ✅ [PASS] Heatmap peaks look healthy (> 0.3).")

    print("\n🔍 Keypoint Detail:")
    valid_kpts = 0
    for i in range(17):
        hm = pose_map[i]
        min_v, max_v, _, _ = cv2.minMaxLoc(hm)
        status = "OK" if max_v > 0.1 else "WEAK"
        if max_v > 0.1: valid_kpts += 1
        if i < 5: # 머리 부분만 로그 출력 (너무 길어지니)
            print(f"   - KP {i}: Max={max_v:.4f} [{status}]")
    print(f"   ... (Total Valid Keypoints: {valid_kpts}/17)")

    # 6. Visualization
    # 히트맵 합치기 (모든 관절)
    heatmap_sum = np.sum(pose_map, axis=0)
    # 정규화 (보기 좋게 0~255로)
    heatmap_vis = cv2.normalize(heatmap_sum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    
    # 원본 이미지와 합성
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)
    
    # 세그멘테이션 마스크
    seg_vis = (seg_prob * 255).astype(np.uint8)
    seg_color = cv2.applyColorMap(seg_vis, cv2.COLORMAP_BONE)
    
    # 결과창 띄우기
    cv2.imshow("DIAGNOSIS: Heatmaps (Pose)", overlay)
    cv2.imshow("DIAGNOSIS: Mask (Seg)", seg_color)
    
    print("\n👀 Check the popup windows. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    profile = sys.argv[1] if len(sys.argv) > 1 else "front"
    debug_weights(profile)