# 🏭 제조업 안전사고 감지 시스템 - Streamlit 데모  

학습된 CNN-LSTM 모델을 사용하여 제조업 안전사고를 실시간으로 감지하는 웹 데모 애플리케이션입니다.  

## 🚀 실행 방법  

### 1. 의존성 설치  
```bash
# uv 에 추가하기  
uv add streamlit  

# uv 환경에서 실행  
cd safety_demo  
# uv pip install -r requirements.txt  

# 또는 pip 사용  
# pip install -r requirements.txt  
```

### 2. 앱 실행  
```bash
uv run streamlit run app.py  
```

### 3. 브라우저에서 접속  
- 자동으로 브라우저가 열리며 `http://localhost:8501`에서 접속 가능합니다.  

## 📋 사용 방법  

1. **모델 로드**: 사이드바에서 학습된 모델을 선택  
   - 사용 가능한 모델 목록에서 드롭다운으로 선택
   - 기본값: `best.pth` (최고 성능 모델)  
   
2. **비디오 업로드**: 분석할 비디오 파일을 업로드  
   - 지원 형식: MP4, AVI, MOV, MKV, WMV  
   
3. **분석 실행**: "안전사고 분석 시작" 버튼 클릭  

4. **결과 확인**:  
   - 예측된 사고 유형과 신뢰도  
   - 모든 클래스별 확률 분포  
   - 분석에 사용된 샘플 프레임들  

## 🎯 감지 가능한 사고 유형  

- **충돌 사고** (bump)  
- **넘어짐 사고** (fall-down)  
- **추락 사고** (fall-off)  
- **타격 사고** (hit)  
- **끼임 사고** (jam)  
- **정상 상황** (no-accident)  

## 🔧 기술 스택  

- **Frontend**: Streamlit  
- **ML Framework**: PyTorch  
- **Computer Vision**: OpenCV, torchvision  
- **Model**: CNN-LSTM (ResNet-101 + LSTM)  

## 📝 주의사항  

- 모델 파일(`best.pth`)이 존재해야 정상 작동합니다.  
- 학습이 완료된 후 `learn_deeplearning/snapshots/` 폴더에서 모델 파일을 확인하세요.
- 경로 설정은 `config.py` 파일에서 쉽게 변경할 수 있습니다.  
- GPU가 있으면 자동으로 GPU를 사용하며, 없으면 CPU로 동작합니다.  

## 🐛 문제 해결  

### 모델 파일을 찾을 수 없는 경우  
- 학습이 완료되었는지 확인  
- `snapshots/best.pth` 파일 존재 여부 확인  
- 사이드바에서 올바른 경로 설정  

### 비디오 업로드 오류  
- 지원되는 형식인지 확인 (MP4, AVI, MOV, MKV, WMV)  
- 파일 크기가 너무 크지 않은지 확인 (권장: 100MB 이하)  

### 메모리 부족 오류  
- 비디오 길이가 너무 긴 경우 발생 가능  
- 짧은 비디오(30초 이하)로 테스트 권장  

## ⚙️ 설정 파일 (config.py)

경로 및 설정값들을 `config.py`에서 쉽게 변경할 수 있습니다:

```python
# 주요 경로 설정
PROJECT_ROOT = Path(r"C:\Users\bb\Desktop\learn_deeplearning")
MODEL_SNAPSHOTS_DIR = PROJECT_ROOT / "snapshots"
DEFAULT_MODEL_PATH = MODEL_SNAPSHOTS_DIR / "best.pth"

# 모델 설정
NUM_CLASSES = 6
FIXED_SEQUENCE_LENGTH = 16
TARGET_FPS = 30
INPUT_SIZE = (224, 224)
```

**경로 변경 방법:**
1. `config.py` 파일 열기
2. `PROJECT_ROOT` 경로를 본인 환경에 맞게 수정
3. 앱 재시작  
