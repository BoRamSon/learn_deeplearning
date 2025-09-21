# 🟩 Learn DeepLearning for **Industrial accident**   
- Start project on August 26, 2025  
- Ended project on September 21, 2025  


<br>

## 📁 Repository Structure  
📦 **Industrial accident**  
┣ 📂 [Book]Deep_Learning_from_Scratch `(공부)`  
┃ ┗ 📂 chapter 01 ~ 08  
┃  
┣ 📂 backend `(❌ 웹 구축 시도 후 가중치 업로드 실패)`  
┃ ┣ 📂 snapshots (Included in the gitignore)  
┃ ┃ ┗ 📄 best.pth (183MB)  
┃  
┣ 📂 frontend `(❌ 웹 구축 시도 후 가중치 업로드 실패)`  
┃ ┗ 📂 src  
┃  
┣ 📂 human-accident `(Computer Vision Project 'Prevention of occupational fatalities')`  
┃ ┣ 📂 [cnn-lstm](https://github.com/pranoyr/cnn-lstm) `(clone하여 가져온 모델)`  
┃ ┣ 📂 data `(산업재해 관련 동영상 데이터)`  
┃ ┗ 📄 data_and_label.ipynb ~ my_custom_dataset.ipynb  
┃  
┣ 📂 Record `(공부과정 기록)`  
┃ ┣ 📄 과정 정리  
┃ ┗ 📄 발표목차  
┃  
┣ 📂 Record_Blog `(Blog 내용 정리)`  
┃ ┣ 📄 20250909.ipynb  
┃ ┗ 📄 20250909.md  
┃  
┣ 📂 safety_demo `(streamlit)`  
┃ ┣ 📂 sample_videis `(산업재해 관련 동영상 샘플)`  
┃ ┗ 📄 (streamlit 관련 파일들)  
┃  
┣ 📂 snapshots (Included in the gitignore)  
┃ ┗ 📄 best.pth (183MB) `(학습된 모델)`  
┃  
┣ 📄 .gitignore  
┣ 📄 .python-version  
┣ 📄 main.py  
┣ 📄 pyproject_toml.md  
┣ 📄 pyproject.toml  
┣ 📄 README.md  
┗ 📄 uv.lock  

<br><br>

## ✏️ Study with **`[Book]DeepLearning from Scratch`**  
- Chapter 01 : ██████████ 100%  
- Chapter 02 : ██████████ 100%  
- Chapter 03 : ████████░░ 80%  
- Chapter 04 : ░░░░░░░░░░ 0%  
- Chapter 05 : ░░░░░░░░░░ 0%  
- Chapter 06 : ░░░░░░░░░░ 0%  
- Chapter 07 : ░░░░░░░░░░ 0%  
- Chapter 08 : ░░░░░░░░░░ 0%  


<br><br>

## 📜 Computer Vision Project **`[Prevention of occupational fatalities]`**  

### 🟡 데이터셋: 스마트 제조 시설 안전 감시를 위한 데이터  
- 데이터셋 출처: [스마트 제조 시설 안전 감시를 위한 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EC%8A%A4%EB%A7%88%ED%8A%B8%EC%A0%9C%EC%A1%B0&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71679)  

### 🟡 학습 모델: CNN-LSTM  
- 모델 출처: https://github.com/pranoyr/cnn-lstm  


### 🟡 결과 (best.pth)  
- [Epoch 10/10] train_loss=1.3484 train_acc=40.00% | val_loss=1.7288 val_acc=31.48%  



<br><br>

## ❌ Website for Prevention of occupational fatalities Project  
- ⭕️ frontend: Next.js  
    - install node.js  
    - cd frontend  
    - npm install  
    - npm run dev  


- ⭕️ backend: FastAPI  
    - cd backend  
    - pip install -r requirements.txt  
    - uvicorn main:app --host 0.0.0.0 --port 8000 --reload  


- ❌ render.com (가중치 실패)  
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
                - Environment Variables: PYTHON_VERSION=3.11.0  
                - Backend Deployment successful: https://safety-detection-backend.onrender.com  

            - frontend  
                - Name: safety-detection-frontend  
                - Environment: Node  
                - Region: Oregon (US West)  
                - Branch: main  
                - Root Directory: frontend  
                - Build Command: npm install; npm run build  
                - Start Command: npm start  
                - Environment Variables: NEXT_PUBLIC_API_URL=https://safety-detection-backend.onrender.com  (백엔드 URL)  
                - Frontend Deployment successful: https://safety-detection-frontend.onrender.com  


<br><br>

## 🏆 Work Toward a Goal  
- [ ] None  
- [ ] None  


<br><br>

## ➡️ How to 'Git Collaboration Works'  

- (locked main branch)  
- Create Issue  
  - Organization -> Repository -> Project -> `add item`  
- Create Issue Branch  
  - Organization -> Repository -> Project -> Select item -> Development -> Create a Baranch -> Setting and `Create branch`  
- Check Issue  
  - Organization -> Repository -> Issue  
- Coding for the Issue  
- And `Pull requests` to `'feat' branch`  
- All Team members `CodeReview` on the `Pull requests`  
- It can merge to `'dev' branch` after approved the 'feature code'  
- Repeat the above process  
- Done.  


<br><br>

## 📝 History  
 

<br><br>

## 🔗 reference  





