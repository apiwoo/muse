import os
import shutil

# ==========================================
# [수정] Project MUSE 개발을 위한 확장자 및 파일명 설정
# 핵심: 쉐이더(.glsl)와 UI 스타일(.qss)이 포함되어야 합니다.
ALLOWED_EXTENSIONS = {
    # Backend / Core Logic
    ".py", ".sql", ".ini", ".conf",
    
    # Frontend / UI / Graphics (중요)
    ".dart", ".yaml", ".json", ".xml",
    ".html", ".js", ".css",
    ".qss",   # [추가] PyQt 스타일시트
    ".glsl",  # [추가] OpenGL 쉐이더 코드 (핵심 그래픽 로직)
    
    # Config / Documentation
    ".md", ".txt"
}

# [추가] 확장자로 판별하기 어려운 특수 파일명들 (정확히 일치하면 복사)
# 기존 코드의 잠재적 버그(확장자 인식 오류)를 해결하기 위해 추가함
ALLOWED_FILENAMES = {
    ".gitignore", "Dockerfile", "docker-compose.yml",
    "requirements.txt", "Pipfile", ".env", "Makefile"
}

# [설정] 결과물이 저장될 폴더명 (압축파일 아님)
OUTPUT_DIR_NAME = "core_source_code_folder"

# [설정] 탐색하지 않을 폴더들 (Project MUSE 구조 반영)
IGNORE_FOLDERS = {
    # 시스템 및 IDE 설정
    ".git", ".idea", ".vscode", "venv", "__pycache__", 
    "build", "dist", ".dart_tool", ".gradle", "node_modules",
    
    # OS 관련
    "ios", "android", "linux", "macos", "windows", 
    
    # [수정] assets 폴더 전체 제외 제거 -> 세부 폴더 제외로 변경
    # "assets",  <-- (삭제됨) 이제 assets 폴더 내부를 탐색합니다.
    
    # [추가] 대용량 바이너리나 미디어 리소스만 콕 집어서 제외
    "models",    # AI 가중치 파일 (.pth 등) - 용량이 커서 코드 공유 시 제외
    # "images",  # UI 아이콘만 제외하려 했으나, 라벨링 결과(images)도 이름이 같으므로 여기서 제외됨
    "videos",    # 테스트용 비디오
    "fonts",      # 폰트 파일
    
    # [NEW] 학습 데이터 및 라벨링 결과물 제외 (압축 대상 아님)
    "recorded_data", # 녹화된 원본 영상 및 데이터 루트
    "images",        # 라벨링된 이미지 (recorded_data 내부에 있지만 안전장치로 추가)
    "masks",         # 라벨링된 마스크
    "labels",        # 라벨링된 JSON 데이터
    
    # [NEW] 자기 자신 제외 (재귀 복사 방지)
    OUTPUT_DIR_NAME
}
# ==========================================

def copy_core_code():
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, OUTPUT_DIR_NAME)
    
    # [중요] 무한 루프 방지: 결과 폴더가 탐색 대상에 포함되지 않도록 제외 목록에 추가
    IGNORE_FOLDERS.add(OUTPUT_DIR_NAME)
    
    file_count = 0
    
    print(f"🚀 Project MUSE 핵심 소스코드 추출(복사) 시작: {current_dir}")
    print(f"📂 대상 폴더: {OUTPUT_DIR_NAME} (덮어쓰기 모드)")
    print(f"ℹ️  설정: 확장자 {len(ALLOWED_EXTENSIONS)}종, 특수파일 {len(ALLOWED_FILENAMES)}종 포함")
    print(f"🚫 제외 폴더: recorded_data, masks, labels, images 등")

    # 결과 폴더가 없으면 생성 (있으면 무시)
    os.makedirs(target_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(current_dir):
        # 제외 폴더는 아예 진입하지 않음 (리스트를 직접 수정하여 os.walk 제어)
        # 여기서 assets가 빠졌으므로, assets 폴더로 진입하되 models 등은 걸러짐
        dirs[:] = [d for d in dirs if d not in IGNORE_FOLDERS]
        
        for file in files:
            is_target = False
            
            # 1. 파일명 일치 여부 확인 (우선순위 높음)
            if file in ALLOWED_FILENAMES:
                is_target = True
            else:
                # 2. 확장자 체크 (허용된 것만 통과)
                _, ext = os.path.splitext(file)
                if ext.lower() in ALLOWED_EXTENSIONS:
                    is_target = True
            
            # 대상이 아니면 건너뜀
            if not is_target:
                continue
            
            # 3. 파일명 예외 처리 (이 스크립트 파일 자체는 제외)
            if file == "compress_project.py":
                continue

            # 원본 파일 경로
            src_path = os.path.join(root, file)
            
            # 상대 경로 계산 (프로젝트 루트 기준)
            rel_path = os.path.relpath(src_path, current_dir)
            
            # 대상 파일 경로 (결과 폴더 내)
            dst_path = os.path.join(target_dir, rel_path)
            
            # 대상 파일의 상위 폴더가 없으면 생성
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # 파일 복사 (메타데이터 유지, 에러 처리 추가)
            try:
                shutil.copy2(src_path, dst_path)
                file_count += 1
            except Exception as e:
                print(f"❌ 복사 실패: {rel_path} - {e}")
            
    print(f"\n✅ 추출 완료! (총 {file_count}개 파일)")
    print(f"👉 파일들이 '{OUTPUT_DIR_NAME}' 폴더에 복사되었습니다.")
    print(f"👉 이 폴더를 압축하거나 AI에게 공유하여 개발을 진행하세요.")

if __name__ == "__main__":
    copy_core_code()