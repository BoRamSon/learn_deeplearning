from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
import tempfile
import os
import uuid
from typing import Dict
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

# 백그라운드 작업 결과를 저장할 딕셔너리
tasks: Dict[str, Dict] = {}

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
def load_model():
    """서버 시작 시 모델을 로드합니다."""
    global model
    if model is None: # 중복 로딩 방지
        try:
            print("Loading model on demand...")
            model = CNNLSTM(num_classes=6)
            
            model_paths = [
                "snapshots/best.pth", # Render.com 배포 환경에서 일반적인 경로
                "best.pth", # 로컬 테스트용
                "../snapshots/best.pth" # 다른 경로 구조일 경우 대비
            ]
            
            model_loaded = False
            for path in model_paths:
                if os.path.exists(path):
                    print(f"Model loaded from: {path}")
                    # checkpoint는 'state_dict' 키를 포함하는 딕셔너리입니다.
                    checkpoint = torch.load(path, map_location=device)
                    # state_dict를 직접 로드해야 합니다.
                    model.load_state_dict(checkpoint['state_dict'])
                    model_loaded = True
                    break
            
            if not model_loaded:
                print("Model file not found. Using dummy model.")
            
            model.to(device)
            model.eval()
            print("Model loading complete!")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            model = None

def process_video(video_path, fixed_len=16):
    """비디오 전처리"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
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

def run_prediction_in_background(file_content: bytes, task_id: str):
    """백그라운드에서 비디오 처리 및 추론을 수행하는 함수"""
    tasks[task_id] = {"status": "processing", "result": None}
    try:
        # 임시 파일에 비디오 내용 쓰기
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(file_content)
            temp_path = tmp_file.name

        # 비디오 처리 (CPU 집약적)
        video_tensor = process_video(temp_path)
        os.unlink(temp_path)

        if video_tensor is None:
            raise ValueError("Failed to process video. It might be empty or corrupted.")

        # 추론 (CPU 집약적)
        with torch.no_grad():
            video_tensor = video_tensor.to(device)
            logits = model(video_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu()
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        result = {
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
                for i, prob in enumerate(probabilities[0].numpy())
            }
        }
        tasks[task_id] = {"status": "completed", "result": result}

    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        tasks[task_id] = {"status": "failed", "result": str(e)}

@app.post("/predict", status_code=202)
async def predict_async(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """비디오를 업로드하고 백그라운드 처리를 시작합니다."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")
    
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Only video files are supported.")

    task_id = str(uuid.uuid4())
    file_content = await file.read()

    # 백그라운드 작업 추가
    background_tasks.add_task(run_prediction_in_background, file_content, task_id)

    return {
        "message": "Video upload successful. Processing has started.",
        "task_id": task_id,
        "status_url": f"/tasks/{task_id}"
    }

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """작업 ID로 처리 상태와 결과를 조회합니다."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    
    if task["status"] == "completed":
        # 성공적으로 완료된 작업은 메모리에서 제거할 수 있습니다.
        # 여기서는 간단히 유지하지만, 실제 서비스에서는 TTL(Time-To-Live)을 두는 것이 좋습니다.
        pass

    return task

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
