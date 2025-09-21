# 🟩 모델 학습  

본 문서는 `human-accident` 프로젝트에서 커스텀 비디오 데이터로 CNN-LSTM을 학습할 수 있도록 구성한 전반적인 과정을 정리  

<br>

## 📋 프로젝트 개요  
- **목표**: 제조업 안전사고 예방을 위한 컴퓨터 비전 모델 개발  
- **데이터셋**: AI허브 "스마트 제조 시설 안전 감시를 위한 데이터"  
- **모델**: CNN-LSTM (pranoyr/cnn-lstm 저장소 활용)  
- **개발환경**: Windows, NVIDIA GeForce GTX 1650, Python 3.11, uv 패키지 관리자  

<br>

## 🟢 1) 데이터셋 구조와 경로  
```
    human-accident/  
    ├── data/  
    │   └── safety-data/  
    │       └── human-accident/  
    │           ├── bump/           # 충돌 사고 비디오 (136개 파일)  
    │           ├── fall-down/      # 넘어짐 사고  
    │           ├── fall-off/       # 추락 사고  
    │           ├── hit/            # 타격 사고  
    │           ├── jam/            # 끼임 사고  
    │           └── no-accident/    # 정상 상황  
    │  
    ├── cnn-lstm/                   # 클론된 모델 저장소  
    │   ├── main.py                 # 메인 훈련 스크립트  
    │   ├── opts.py                 # 커맨드라인 옵션 정의  
    │   ├── models/cnnlstm.py       # CNN-LSTM 모델 아키텍처  
    │   ├── dataset.py              # 데이터셋 로더  
    │   └── datasets/ucf101.py      # UCF101 데이터셋 클래스  
    │  
    │  
    │  
    └── 30_model_training.ipynb  
```
- **데이터 루트**: `human-accident/data/safety-data/human-accident/`  
- **클래스 폴더(6개)**:  
  - `bump/`, `fall-down/`, `fall-off/`, `hit/`, `jam/`, `no-accident/`  
- 각 클래스 폴더 하위에 비디오 파일(`.avi`, `.mp4`, `.mov`, `.mkv`, `.wmv`)이 위치합니다.  

<br>

## 🟢 2) 커스텀 데이터셋 `CustomData`  
- **파일**: `human-accident/cnn-lstm/train_valid_dataset.py`  
- **역할**:  
  - 루트 경로 하위의 클래스 폴더를 스캔하여 비디오 경로 수집  
  - 클래스별로 셔플 후 **train/valid 70:30** 분할 (시드 고정으로 재현성 보장)  
  - 각 비디오를 불러 **FPS 30**으로 리샘플링 (시간 축 표준화)  
  - 모든 샘플을 **고정 길이 16프레임**으로 맞춤  
    - 길면 균등 샘플링으로 16개 선택  
    - 짧으면 마지막 프레임 반복 패딩으로 16개 확장  
  - 프레임별로 **BGR→RGB 변환**, transform(예: `ToPILImage→Resize(224,224)→ToTensor→Normalize`) 적용  
  - 최종 출력 텐서 형태: **(T, C, H, W)**  
  - 레이블은 클래스 인덱스(int)  

<br>

## 🟢 3) 데이터로더 구성  
- **함수**: `get_dataloaders()`  
- **파일**: `human-accident/cnn-lstm/main.py`  
- **내용**:  
  - 공통 transform 구성: `ToPILImage→Resize(224,224)→ToTensor→Normalize`  
  - `CustomData`로 train/valid 데이터셋 생성  
  - `DataLoader`로 래핑 (train은 shuffle=True, valid는 shuffle=False)  
  - 배치 입력 형태: **(B, T, C, H, W)**  

<br>

## 🟢 4) 모델  
- **파일**: `human-accident/cnn-lstm/models/cnnlstm.py`  
- **클래스**: `CNNLSTM(num_classes=6)`  
  - 프레임별 특징 추출: 사전학습된 **ResNet-101**(fc 교체 → 300차원)  
  - 시퀀스 모델링: **LSTM(입력 300, hidden 256, num_layers=3)**  
  - 분류기: `256 → 128 → num_classes`  
  - 입력은 (B, T, C, H, W)이며, 내부에서 T 타임스텝 순회로 처리  
```python
    class CNNLSTM(nn.Module):  
        def __init__(self, num_classes=2):  
            # ResNet-101 (사전훈련) → 300차원 특징 추출  
            # torchvision 최신 권장 API: pretrained 대신 weights enum 사용  
            from torchvision.models import resnet101, ResNet101_Weights  
            self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)  
            self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))  
            
            # LSTM: 300 → 256 (3 레이어)  
            self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)  
            
            # 분류기: 256 → 128 → num_classes  
            self.fc1 = nn.Linear(256, 128)  
            self.fc2 = nn.Linear(128, num_classes)  
```

<br>

## 🟢 5) 학습 엔트리: `main.py`  
- **파일**: `human-accident/cnn-lstm/main.py`  
- **구성 요약**:  
  - argparse로 하이퍼파라미터 수집  
  - `get_dataloaders()`로 train/valid 로더 생성  
  - 모델/손실/옵티마이저 정의  
    - `model = CNNLSTM(num_classes=6)`  
    - `criterion = nn.CrossEntropyLoss()`  
    - `optimizer = Adam(model.parameters(), lr=..., weight_decay=...)`  
  - 에폭 루프: train/valid 실행, 로그 출력  
  - 매 에폭 체크포인트 저장: `snapshots/CNNLSTM-epoch{epoch}-valacc{...}.pth`  
  - 최고 성능(best) 모델 저장: `snapshots/best.pth`  
  - (옵션) `--use-scheduler` 플래그로 **ReduceLROnPlateau** 스케줄러 사용 가능 (기본 비활성화)  

<br>

## 🟢 6) 실행 방법 (예시)  
- uv 환경에서 실행:  
```bash
uv run python human-accident/cnn-lstm/main.py \  
  --root ../data/safety-data/human-accident \  
  --epochs 5 \  
  --batch-size 8 \  
  --lr 1e-4 \  
  --weight-decay 1e-4 \  
  --num-workers 2 \  
  --train-ratio 0.7 \  
  --seed 42 \  
  --num-classes 6  
```
- 학습률 스케줄러 활성화:  
```bash
uv run python human-accident/cnn-lstm/main.py --use-scheduler  
```

추가 실행 팁(Windows PowerShell):  
- 여러 줄 명령은 줄 끝에 캐럿(^) 또는 백틱(`)을 사용하세요. 줄 앞에 `+` 같은 문자가 있으면 파싱 오류가 납니다.  
- 상대경로 대신 절대경로 사용을 권장합니다. `main.py`는 `--root`를 절대경로로 변환하여 출력합니다.  
  - 예: `--root "C:\\Users\\bb\\Desktop\\learn_deeplearning\\human-accident\\data\\safety-data\\human-accident"`  

<br>

## 🟢 7) 핵심 포인트 정리  
- **시간 축 표준화**: `same_fps()`로 FPS=30 고정 → 시퀀스 샘플링의 시간적 의미를 일관되게 유지  
- **길이 고정(16프레임)**: 균등 샘플링/패딩으로 배치 텐서 생성 가능 (모든 샘플 길이 동일)  
- **프레임 전처리 표준화**: BGR→RGB, PIL 변환, Resize(224), Normalize(ImageNet)  
- **간단 학습 진입점**: 기존 UCF101 전용 코드 제거, 커스텀 데이터셋 중심으로 재구성  
- **체크포인트 관리**: 매 에폭 저장 + best.pth 별도 보관  

추가 개선 사항(최근 반영):  
- **절대 경로 해석**: `main.py`에서 `--root`를 `Path.resolve()`로 절대경로화하여 경로 혼동 방지 및 친절한 오류 메시지 출력.  
- **tqdm 진행바**: 학습/검증 루프에 tqdm 적용, 러닝 평균 `loss/acc`를 postfix로 표시.  
- **GPU 디버그 로그**: 첫 학습 배치에서 입력/모델/LSTM/ResNet의 device, CUDA 이름, 메모리 사용량을 출력해 GPU 사용 여부를 즉시 확인.  
- **torchvision 경고 제거**: `pretrained=True` 대신 `weights=ResNet101_Weights.DEFAULT` 사용.  
- **디코딩 경고 가이드**: MPEG4 디코딩 경고(`ac-tex damaged`, `Error at MB`) 발생 시, ffmpeg 리믹스/재인코딩 방법 안내.  

<br>

## 🟢 8) 다음 확장 아이디어  
- EarlyStopping 플래그 추가 (val_acc 개선 없으면 중단)  
- 클래스 불균형 시 가중치 적용 (CrossEntropyLoss에 class weight)  
- 입력 해상도(224→112) 또는 길이(16→8/32) 변경으로 자원 최적화  
- 추론/평가 스크립트 추가 (best.pth 로드 후 단일 비디오/폴더 평가)  
- WandB나 TensorBoard 로깅 연결(원할 때 선택적으로 추가)  

<br>

이 문서는 현재 동작 파이프라인 기준으로 작성되었습니다. 필요한 변경/확장 사항이 생기면 본 문서를 갱신해 주세요.  

---

## 🟡 부록 A. 디코딩 경고(FFmpeg) 대응 가이드  
- 증상 예시: `ac-tex damaged at X Y`, `Error at MB: Z` (MPEG4 디코딩 중)  
- 원인: 손상/불완전 인코딩, 컨테이너/코덱 문제 등으로 특정 매크로블록을 해석 못함  
- 영향: 일부 프레임 읽기 실패 → 더미 프레임 처리로 학습은 진행되나 품질 저하 가능  
- 대응:  
  - 무손실 리믹스(빠름):  
    ```powershell
    ffmpeg -err_detect ignore_err -fflags +discardcorrupt -i "input.mp4" -c copy "remux.mp4"  
    ```
  - 안전 재인코딩(H.264, 30fps):  
    ```powershell
    ffmpeg -v error -err_detect ignore_err -fflags +genpts -i "input.mp4" -c:v libx264 -pix_fmt yuv420p -r 30 -an "reencoded.mp4"  
    ```
  - 에러 탐지만:  
    ```powershell
    ffmpeg -v error -i "input.mp4" -f null - 2>errors.log  
    ```

## 🟡 부록 B. GPU 사용 확인 방법  
- 학습 시작 시 첫 배치에 다음 정보가 콘솔에 출력됩니다.  
  - 입력 텐서 `x.device`, 모델 파라미터 device, ResNet fc device, LSTM device  
  - `CUDA name`, `CUDA memory allocated (MB)`  
- 예상 출력: 모두 `cuda:0`이면 GPU 사용 중. `cpu`로 나오면 CUDA 설정/설치 점검 필요.  


---
# 🟩 학습 결과  
- 25.09.17  
  - [Epoch 10/10] train_loss=1.3484 train_acc=40.00% | val_loss=1.7288 val_acc=31.48% | time=2272.3s  