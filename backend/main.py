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
        model = CNNLSTM(num_classes=6)
        
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
                checkpoint = torch.load(path, map_location=device)
                model.load_state_dict(checkpoint)
                model_loaded = True
                break
        
        if not model_loaded:
            print("Model file not found. Using dummy model.")
        
        model.to(device)
        model.eval()
        print("Model loading complete!")
        
    except Exception as e:
        print(f"Model loading failed: {e}")

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
    return video_tensor

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
            video_tensor = video_tensor.to(device)
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
