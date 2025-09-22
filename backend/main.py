from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
import tempfile
import os
from torchvision import transforms
from model import CNNLSTM

app = FastAPI(title="Safety Detection API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU 디버깅 정보 출력
print("🔍 GPU/디바이스 정보:")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"GPU 개수: {torch.cuda.device_count()}")
    print(f"현재 GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("❌ CUDA를 사용할 수 없습니다.")
    print("가능한 해결 방법:")
    print("1. PyTorch CUDA 버전 설치")
    print("2. NVIDIA 드라이버 확인")
    print("3. CUDA toolkit 설치")
    print("4. nvidia-smi 명령어로 GPU 상태 확인")

print(f"선택된 디바이스: {device}")

# 클래스 정의
CLASS_NAMES = ["bump", "fall-down", "fall-off", "hit", "jam", "no-accident"]
CLASS_NAMES_KR = ["충돌", "넘어짐", "추락", "타격", "끼임", "정상"]

# Transform 정의
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.on_event("startup")
async def startup_event():
    global model
    try:
        print("Loading model...")

        # 먼저 device 확인
        print(f"Using device: {device}")

        # 모델을 device에서 생성
        model = CNNLSTM(num_classes=6).to(device)

        # 모델 파일 경로들 시도
        model_paths = [
            "best.pth",
            "../snapshots/best.pth",
            "snapshots/best.pth"
        ]

        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                print(f"Model loaded from: {path}")
                # 가중치를 로드할 때도 device 명시
                checkpoint = torch.load(path, map_location=device)
                model.load_state_dict(checkpoint)
                model_loaded = True
                break

        if not model_loaded:
            print("Model file not found. Using dummy model.")

        # 모델이 이미 device에 있으므로 다시 to(device) 호출 불필요
        model.eval()
        print("Model loading complete!")

    except Exception as e:
        print(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()

def process_video(video_path, fixed_len=16):
    """비디오 전처리"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        return None

    # 16프레임으로 맞추기
    if len(frames) >= fixed_len:
        indices = np.linspace(0, len(frames) - 1, fixed_len, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        while len(frames) < fixed_len:
            frames.append(frames[-1])

    # Transform 적용
    processed_frames = []
    for frame in frames:
        tensor_frame = transform(frame)
        processed_frames.append(tensor_frame)

    video_tensor = torch.stack(processed_frames).unsqueeze(0)  # (1, T, C, H, W)
    return video_tensor.to(device)  # GPU로 이동

@app.get("/")
async def root():
    return {"message": "Safety Detection API", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="비디오 파일만 지원합니다")

    try:
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name

        # 비디오 처리
        video_tensor = process_video(temp_path)
        os.unlink(temp_path)  # 임시 파일 삭제

        if video_tensor is None:
            raise HTTPException(status_code=400, detail="비디오 처리 실패")

        # 예측
        with torch.no_grad():
            # video_tensor는 이미 GPU에 있으므로 추가 .to(device) 불필요
            logits = model(video_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        return {
            "success": True,
            "prediction": {
                "class_id": predicted_class,
                "class_name": CLASS_NAMES[predicted_class],
                "class_name_kr": CLASS_NAMES_KR[predicted_class],
                "confidence": float(confidence),
                "is_accident": predicted_class != 5
            },
            "probabilities": {
                CLASS_NAMES_KR[i]: float(prob)
                for i, prob in enumerate(probabilities[0].cpu().numpy())
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)