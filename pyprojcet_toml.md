## 🔍 torch 설치 확인 (진짜 확인하려면 c++ 관련 프로그램 설치 필요)  

<br>

---

## 🟢 Windows/Linux (CUDA 12.8):  
uv run python -c "import torch, torchvision, torchaudio; print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"  

--- 
### Windows에서 실행한 결과  
uv run python -c "import torch, torchvision, torchaudio; print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"  

#### 출력 내용  
torch: 2.8.0+cu128, CUDA: True  

이런 결과가 나왔다.  
- ✅ 2.8.0+cu128 → PyTorch CUDA 12.8 빌드가 정확히 설치됨  
- ✅ CUDA: True → NVIDIA GPU를 정상적으로 인식해서 GPU 학습 환경이 준비 완료됨  



<br><br>

---
---


## 🟢 macOS (M1/M2/M3 → MPS):  
uv run python -c "import torch, torchvision, torchaudio; print(f'torch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"  