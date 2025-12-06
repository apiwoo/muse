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
    ".qss",   # PyQt 스타일시트
    ".glsl",  # OpenGL 쉐이더 코드
    
    # Config / Documentation
    ".md", ".txt"
}

# 확장자로 판별하기 어려운 특수 파일명들
ALLOWED_FILENAMES = {
    ".gitignore", "Dockerfile", "docker-compose.yml",
    "requirements.txt", "Pipfile", ".env", "Makefile"
}

# [설정] 결과물이 저장될 폴더명
OUTPUT_DIR_NAME = "core_source_code_folder"

# [설정] 탐색하지 않을 폴더들 (Project MUSE 구조 반영)
IGNORE_FOLDERS = {
    # 시스템 및 IDE 설정
    ".git", ".idea", ".vscode", "venv", "__pycache__", 
    "build", "dist", ".dart_tool", ".gradle", "node_modules",
    
    # OS 관련
    "ios", "android", "linux", "macos", "windows", 
    
    # 리소스 제외
    "models",    # AI 가중치 파일
    "videos",    # 테스트용 비디오
    "fonts",     # 폰트 파일
    "libs",      # DLL 라이브러리 폴더 (코드 아님)
    
    # 데이터 제외
    "recorded_data", 
    "images",        
    "masks",         
    "labels",        
    
    # [NEW] 개인화 학습(Personalized Learning) 관련 레거시 폴더 제외
    "distillation",  # 학습 로직 (Teacher/Student)
    "auto_labeling", # 라벨링 도구
    "studio",        # 학습용 GUI 패키지 (Wizard)
    
    # 자기 자신 제외
    OUTPUT_DIR_NAME
}

# [NEW] 제외할 특정 파일명 (레거시/실험용/문서 파일)
IGNORE_FILES = {
    # 문서 (이제 불필요함)
    "개인학습가이드.txt",        # [삭제] 개인화 학습 가이드

    # 학습 및 데이터 수집 도구 (레거시)
    "recorder.py",              # 데이터 녹화
    "train_student.py",         # 학생 모델 학습
    "convert_student_to_trt.py",# 학생 모델 변환
    "muse_studio.py",           # 학습용 GUI 런처
    
    # 구버전/미사용 로직
    "body_tracker.py",          # 구버전 개인화 모델 로더
    "personalizer.py",          # 미사용 클래스
    "validator.py",             # 내용 없는 파일
    
    # 유틸리티 (필요시 제외 해제 가능)
    "collect_libs.py",          # 배포용 툴
    "clean_launch.py",          # 캐시 정리 툴
    "fix_env.py",               # 환경 복구 툴
    "patch_dll.py",             # DLL 패치 툴
    "force_patch_ort.py",       # ONNX 패치 툴
    "check_gpu.py",             # GPU 확인 툴
    "download_models.py"        # 모델 다운로더 (코드가 아닌 유틸리티)
}

# [NEW] 제외할 파일명 패턴 (접두사)
IGNORE_PREFIXES = (
    "test_",    # 각종 단위 테스트 (test_sam.py, test_mediapipe.py 등)
    "debug_",   # 디버깅 스크립트 (debug_pipeline.py 등)
    "compare_"  # 비교 실험 스크립트 (compare_pose_hmr.py 등)
)

# ==========================================

def copy_core_code():
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, OUTPUT_DIR_NAME)
    
    # 무한 루프 방지
    IGNORE_FOLDERS.add(OUTPUT_DIR_NAME)
    
    file_count = 0
    excluded_count = 0
    
    print(f"🚀 Project MUSE 핵심 소스코드 추출(복사) 시작: {current_dir}")
    print(f"📂 대상 폴더: {OUTPUT_DIR_NAME} (덮어쓰기 모드)")
    
    # 결과 폴더 생성
    os.makedirs(target_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(current_dir):
        # 1. 폴더 필터링
        dirs[:] = [d for d in dirs if d not in IGNORE_FOLDERS]
        
        for file in files:
            # 2. 제외 파일명 체크
            if file in IGNORE_FILES:
                excluded_count += 1
                continue
                
            # 3. 제외 패턴 체크 (Startswith)
            if file.startswith(IGNORE_PREFIXES):
                excluded_count += 1
                continue
                
            # 4. 자기 자신 제외
            if file == "compress_project.py":
                continue

            is_target = False
            
            # 5. 포함 대상 확인 (파일명 or 확장자)
            if file in ALLOWED_FILENAMES:
                is_target = True
            else:
                _, ext = os.path.splitext(file)
                if ext.lower() in ALLOWED_EXTENSIONS:
                    is_target = True
            
            if not is_target:
                continue

            # 복사 수행
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, current_dir)
            dst_path = os.path.join(target_dir, rel_path)
            
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            try:
                shutil.copy2(src_path, dst_path)
                file_count += 1
            except Exception as e:
                print(f"❌ 복사 실패: {rel_path} - {e}")
            
    print(f"\n✅ 추출 완료! (총 {file_count}개 파일 복사됨)")
    print(f"🧹 제외된 레거시/문서/테스트 파일 수: {excluded_count}개")
    print(f"👉 '{OUTPUT_DIR_NAME}' 폴더를 확인하세요.")

if __name__ == "__main__":
    copy_core_code()