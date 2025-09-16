# 🟩 나의 과정  

---
## 🟢 나의 목표는 무엇인가???  

- 산업 안전 관련  

---
## 🟢 데이터 확보하기  

- ㅇㅇ  

---
## 🟢 확보한 데이터 확인 / 간략한 Practice  
1. 데이터 불러오기  
	- 이미지 = 싱글 이미지  
	- 라벨 = 제이슨  

2. 라벨 파일에서 어노테이션 -> 세그멘테이션 라벨을 가져옴  
3. 라벨에서 x, y 포인트를 가져오기  
4. 이미지 위에 점찍기 cv2.circle함수를 이용  

5. 바운딩 박스 그리기 cv2.rectangle  


---
## 🟢 모델 선정하기 커스텀 데이터셋  
- 무슨모델?  
- https://www.kaggle.com/datasets/pypiahmad/realistic-action-recognition-ucf50/data  
- https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition  


--- 
## 🟢 [전처리]  
- 내가 선정한 모델이 어떤 데이터셋을 원하는지 파악하고 거기에 맞게 만들어줘야 합니다.  
    - ucf101 쓰는 것  
    - avi 확장자로 변경하는 것  
    - 폴더 비디오 path 잘 구성해서 폴더에 잘 담아둬야함.  
        - 폴더 이름 class 5개로 해서 하고,  

--- 
## 🟢 [전처리]  
- tensor 형태로 잘 바꿔줘야함  
    - Dataset, DataLoader 에 대해서 공부하기  
- 커스텀 데이터셋 제작.ipynb 파일을 참조하여 공부 잘하기  
    - 유튜브 동영상 잘 참고해보자  
        - 검색어 :  pytorch custom dataset  
        - https://www.youtube.com/watch?v=ZoZHd0Zm3RY  
        - https://www.youtube.com/watch?v=38hn-FpRaJs  


--- 
## 🟢 모델 파악하기  
- git clone https://github.com/pranoyr/cnn-lstm.git 받아왔습니다.  
- 이제 main.py 파일에 주석을 달면서 파악해보겠습니다.  



---
---
--- 
## 🟢 본격적으로 training 들어가기  



