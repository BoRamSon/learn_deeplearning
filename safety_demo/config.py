"""
🔥 Streamlit 데모 앱 설정 파일
경로 및 기타 설정값들을 관리합니다.
"""

import os
from pathlib import Path

# =============================================================================
# 경로 설정
# =============================================================================

# 프로젝트 루트 경로 (learn_deeplearning 폴더)
PROJECT_ROOT = Path(r"C:\Users\bb\Desktop\learn_deeplearning")

# 모델 관련 경로
MODEL_SNAPSHOTS_DIR = PROJECT_ROOT / "snapshots"
DEFAULT_MODEL_PATH = MODEL_SNAPSHOTS_DIR / "best.pth"

# 데이터 관련 경로
DATA_ROOT = PROJECT_ROOT / "human-accident" / "data" / "safety-data" / "human-accident"

# 샘플 비디오 폴더
SAMPLE_VIDEOS_DIR = Path(__file__).parent / "sample_videos"

# =============================================================================
# 모델 설정
# =============================================================================

# 클래스 정의
CLASS_NAMES = [
    "bump",        # 충돌
    "fall-down",   # 넘어짐
    "fall-off",    # 추락
    "hit",         # 타격
    "jam",         # 끼임
    "no-accident"  # 정상
]

CLASS_NAMES_KR = [
    "충돌 사고",
    "넘어짐 사고", 
    "추락 사고",
    "타격 사고",
    "끼임 사고",
    "정상 상황"
]

NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# 비디오 처리 설정
# =============================================================================

# 비디오 전처리 파라미터
FIXED_SEQUENCE_LENGTH = 16  # 고정 프레임 수
TARGET_FPS = 30            # 목표 FPS
INPUT_SIZE = (224, 224)    # 입력 이미지 크기

# 지원하는 비디오 형식
SUPPORTED_VIDEO_FORMATS = ['mp4', 'avi', 'mov', 'mkv', 'wmv']

# ImageNet 정규화 파라미터
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Streamlit 앱 설정
# =============================================================================

# 페이지 설정
PAGE_TITLE = "제조업 안전사고 감지 시스템"
PAGE_ICON = "🏭"

# 파일 업로드 제한
MAX_FILE_SIZE_MB = 100

# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_model_path():
    """기본 모델 경로를 반환합니다."""
    return str(DEFAULT_MODEL_PATH)

def get_snapshots_dir():
    """스냅샷 디렉터리 경로를 반환합니다."""
    return str(MODEL_SNAPSHOTS_DIR)

def get_available_models():
    """사용 가능한 모델 파일 목록을 반환합니다."""
    if not MODEL_SNAPSHOTS_DIR.exists():
        return []
    
    models = []
    for file in MODEL_SNAPSHOTS_DIR.glob("*.pth"):
        models.append({
            'name': file.name,
            'path': str(file),
            'size_mb': round(file.stat().st_size / (1024 * 1024), 1)
        })
    
    # 파일명으로 정렬 (best.pth가 먼저 오도록)
    models.sort(key=lambda x: (x['name'] != 'best.pth', x['name']))
    return models

def validate_paths():
    """중요한 경로들이 존재하는지 확인합니다."""
    paths_status = {
        'project_root': PROJECT_ROOT.exists(),
        'snapshots_dir': MODEL_SNAPSHOTS_DIR.exists(),
        'default_model': DEFAULT_MODEL_PATH.exists(),
        'data_root': DATA_ROOT.exists()
    }
    return paths_status

# =============================================================================
# 개발자용 설정 (필요시 수정)
# =============================================================================

# 디버그 모드
DEBUG = False

# 로깅 레벨
LOG_LEVEL = "INFO"

# GPU 사용 여부 (자동 감지하지만 강제로 CPU 사용하고 싶을 때)
FORCE_CPU = False
