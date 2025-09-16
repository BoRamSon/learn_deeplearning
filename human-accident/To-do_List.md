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
1. 나는 제조업에서 발생하는 안전사고를 사전에 차단하기 위해 컴퓨터 비전 기술을 개발하려고 합니다.  
    - 개발환경  
        - Macbook Pro M1 Pro  
        - astral uv packag 관리자 사용중  
            - 현재 설치된 라이브러리 (현재 다 설치되어 있음.)  
                - [project]  
                    version = "0.1.0"  
                    readme = "README.md"  
                    requires-python = ">=3.11"  

                    dependencies = [  
                        "matplotlib>=3.10.6",  
                        "numpy>=2.3.2",  
                        "torch",  
                        "torchvision",  
                        "torchaudio",  
                        "python-dotenv>=1.1.1",  
                        "opencv-python>=4.11.0.86",  
                        "tensorflow>=2.20.0",  
                        "nbconvert>=7.16.6",  
                        "nbformat>=5.10.4",  
                        "seaborn>=0.13.2",  
                    ]  

                    [[tool.uv.index]]  
                    name = "pytorch-cu128"  
                    url = "https://download.pytorch.org/whl/cu128"  
                    explicit = true  

                    [[tool.uv.index]]  
                    name = "pytorch-cpu"  
                    url = "https://download.pytorch.org/whl/cpu"  
                    explicit = true  

                    # macOS용 MPS 인덱스 추가 (사실상 이것은 cpu와 동일함. 사실 없어도 됨.)  
                    [[tool.uv.index]]  
                    name = "pytorch-mps"  
                    url = "https://download.pytorch.org/whl/cpu"  
                    explicit = true  


                    [tool.uv.sources]  
                    torch = [  
                        # Windows/Linux → CUDA 12.8 wheel 인덱스  
                        { index = "pytorch-cu128", marker = "sys_platform == 'win32' and platform_machine == 'AMD64'" },  
                        { index = "pytorch-cu128", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" },  
                        # macOS → CPU wheel (MPS 지원 포함)  
                        { index = "pytorch-mps", marker = "sys_platform == 'darwin'" },  
                    ]  
                    torchvision = [  
                        # torchvision도 torch와 동일한 인덱스를 사용하도록 설정  
                        { index = "pytorch-cu128", marker = "sys_platform == 'win32' and platform_machine == 'AMD64'" },  
                        { index = "pytorch-cu128", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" },  
                        { index = "pytorch-mps", marker = "sys_platform == 'darwin'" },  
                    ]  
                    torchaudio = [  
                        # torchaudio도 torch와 동일한 인덱스를 사용하도록 설정  
                        { index = "pytorch-cu128", marker = "sys_platform == 'win32' and platform_machine == 'AMD64'" },  
                        { index = "pytorch-cu128", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" },  
                        { index = "pytorch-mps", marker = "sys_platform == 'darwin'" },  
                    ]  


2. AI허브 데이터셋 = 스마트 제조 시설 안전 감시를 위한 데이터  
    - https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EC%8A%A4%EB%A7%88%ED%8A%B8%EC%A0%9C%EC%A1%B0&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71679  
    - 위 데이터셋을 가지고 전처리를 하였으며,  
    - 여기까지 코드를 만들었어요~  
        ```python
            from torch.utils.data import DataLoader  # 데이터로더를 만들기 위함  

            # 이미지에 대한 경로, 일괄적으로 적용할 전처리 이 2가지를 인수로 넣어줌.  
            dataset = CustomData(video_path, transform)  


            dataloader = DataLoader(  
                dataset=dataset,  
                batch_size=8,  # 컴퓨터사양에 따라서 메모리가 처리할 수 있는 양을 정해줘야합니다.  
                # 2n승 으로 늘려주는 편이다. 2,4,8,16,32,64,128,~~256,  
                # 정해주고, 메모리가 감당이 가능한지 확인해봐야합니다.  
                shuffle=True,  
                drop_last=False,  # 마지막에 남는 데이터도 사용합니다. (False: 버리지 않음)  
            )  
        ```


3. 내가 사용할 모델 https://github.com/pranoyr/cnn-lstm 에 대한 코드 파악을 하고, 나의 폴더 내에 통째로 clone을 해둔 상태입니다.  


4. 현재 폴더 구조 정보  
    .venv  
    pyproject.toml  
    uv.lock  
    human-accident/  
    ├── data  
    │   ├──  
    ├── cnn-lstm/                   # pranoyr/cnn-lstm 저장소 clone  
    │   ├── datasets/  
    │   │   └── ucf101.py           # UCF101 데이터셋 로직 정의  
    │   ├── models/  
    │   │   └── cnnlstm.py          # CNN-LSTM 모델 아키텍처 정의  
    │   ├── main.py                 # 전체 학습 파이프라인 실행  
    │   ├── opts.py                 # 커맨드라인 옵션 정의  
    │   ├── dataset.py              # 데이터셋 선택 및 생성  
    │   ├── model.py                # 모델 선택 및 생성  
    │   ├── train.py                # 1 에폭(epoch) 학습 로직  
    │   ├── validation.py           # 1 에폭(epoch) 검증 로직  
    │   ├── mean.py                 # 데이터 정규화를 위한 평균값 계산  
    │   ├── spatial_transforms.py   # 공간적 데이터 증강/변환  
    │   ├── temporal_transforms.py  # 시간적 데이터 증강/변환  
    │   ├── target_transforms.py    # 라벨 데이터 변환  
    │   └── utils.py                # 유틸리티 함수  


5. 저는 이제 해당 모델을 가지고 학습을 시키려고 하는데 어떻게 시키는 것인지 모릅니다... 아주 상세하고 자세하게 차근차근 하나하나 알려주세요!!!  



