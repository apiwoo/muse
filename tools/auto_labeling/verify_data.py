# Project MUSE - verify_data.py
# Data Quality Check Tool
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import cv2
import json
import glob
import numpy as np

def verify_session(session_name):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir, "recorded_data", session_name)
    
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    label_dir = os.path.join(data_dir, "labels")
    
    if not os.path.exists(img_dir):
        print(f"❌ 데이터 폴더를 찾을 수 없습니다: {data_dir}")
        return

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"🔍 검수 시작: {len(img_files)}장")
    print("   [Key] SPACE: 다음 장, B: 뒤로 가기, Q: 종료")
    
    idx = 0
    while idx < len(img_files):
        img_path = img_files[idx]
        basename = os.path.splitext(os.path.basename(img_path))[0]
        
        mask_path = os.path.join(mask_dir, f"{basename}.png")
        label_path = os.path.join(label_dir, f"{basename}.json")
        
        # Load Data
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        with open(label_path, "r") as f:
            label_data = json.load(f)
            keypoints = np.array(label_data["keypoints"])
            
        # Visualization
        # 1. 마스크 오버레이 (빨간색 반투명)
        if mask is not None:
            colored_mask = np.zeros_like(img)
            colored_mask[:, :, 2] = mask # Red Channel
            img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)
            
        # 2. 뼈대 그리기
        for kp in keypoints:
            x, y, conf = kp
            if conf > 0.4:
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
                
        # Info
        cv2.putText(img, f"Frame: {basename} ({idx+1}/{len(img_files)})", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("MUSE Data Verifier", img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'): # Back
            idx = max(0, idx - 1)
        else: # Next (Space or any key)
            idx += 1
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python verify_data.py <SESSION_NAME>")
        sys.exit(1)
        
    verify_session(sys.argv[1])