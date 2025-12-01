# Project MUSE - train_student.py
# (C) 2025 MUSE Corp.
# --------------------------------------------------------
# [사용법]
# python tools/train_student.py <SESSION_NAME>
# 예: python tools/train_student.py 20231025_143000
# --------------------------------------------------------

import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.distillation.trainer import Trainer

def main():
    if len(sys.argv) < 2:
        print("========================================================")
        print("   MUSE Training Launcher")
        print("========================================================")
        print("❌ 세션 이름이 필요합니다.")
        print("   사용법: python tools/train_student.py <SESSION_NAME>")
        print("   예시: python tools/train_student.py 20231025_143000")
        
        # recorded_data 폴더 내의 가능한 세션 목록 표시
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_root = os.path.join(root_dir, "recorded_data")
        if os.path.exists(data_root):
            sessions = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
            if sessions:
                print("\n📂 [가능한 세션 목록]:")
                for s in sorted(sessions):
                    print(f"   - {s}")
        return

    session_name = sys.argv[1]
    
    print("========================================================")
    print(f"   🚀 Training Student Model for Session: {session_name}")
    print("========================================================")
    
    # 학습 설정 (RTX 3060 기준)
    # 메모리가 부족하면 batch_size를 4로 줄이세요.
    trainer = Trainer(session_name, epochs=50, batch_size=8)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n🛑 학습이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()