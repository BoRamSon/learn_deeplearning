# 🟩 learn_deeplearning  

<br>

## 🟢 Study with **`DeepLearning from Scratch`**  


<br><br>

## 🟢 스마트 제조 시설 안전 감시를 위한 데이터  
데이터셋 출처: [스마트 제조 시설 안전 감시를 위한 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EC%8A%A4%EB%A7%88%ED%8A%B8%EC%A0%9C%EC%A1%B0&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71679)  


<br><br>

## 🟢 CNN-LSTM 모델  
- https://github.com/pranoyr/cnn-lstm  




<br><br>

## 🟢 Website for Safety Detection  
- frontend: next.js  
    - cd frontend  
    - npm install  
    - npm run dev  
- backend: fastapi  
    - cd backend  
    - pip install -r requirements.txt  
    - uvicorn main:app --host 0.0.0.0 --port 8000 --reload  
- render  
    - create project  
        - Name: human-accident-project  
            - backend  
                - "New +" → "Web Service" 클릭  
                - Name: safety-detection-backend  
                - Environment: Python 3  
                - Region: Oregon (US West)  
                - Branch: main  
                - Root Directory: backend  
                - Build Command: pip install -r requirements.txt  
                - Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT  
                - PYTHON_VERSION=3.11.0  
            - frontend  
                - "New +" → "Static Site" 클릭  
                - Name: safety-detection-frontend  
                - Environment: Node  
                - Region: Oregon (US West)  
                - Branch: main  
                - Root Directory: frontend  
                - Build Command: npm install && npm run build  
                - Publish Directory: .next  
                - NEXT_PUBLIC_API_URL=https://safety-detection-backend.onrender.com  (백엔드 URL 입력)  