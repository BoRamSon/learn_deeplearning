from customdata import CustomData
import os

from torchvision import transforms
from torch.utils.data import DataLoader

# 데이터로더 만들기
def get_dataloader():
    # --------------------------------
    # 🟩 dataset path 지정
    # 🔥 하드코딩
    origin = "C:\\Users\\bb\\Desktop\\learn_deeplearning\\human-accident\\data\\safety-data\\human-accident\\"
    # origin = "../data/safety-data/human-accident/"
    # origin = "../data/test_out.avi" # mac에서 임시로 path를 지정하였습니다.
    # dataloader.py의 origin은 하드코딩 절대경로입니다. 재현성과 이동성을 위해 아래처럼 바꾸면 좋습니다:
    # origin = os.path.join(os.path.dirname(__file__), "data", "safety-data", "human-accident")

    human_accident_class = os.listdir(origin)
    # print(human_accident_class)

    video_path = []

    for accident_class in human_accident_class:
        for x in os.listdir(origin + accident_class):
            video_path.append(os.path.join(origin, accident_class, x))

    # print(video_path)

    # --------------------------------
    # 🟩 전처리
    # 여러 가지 전처리 방법을 하나로 묶어주는 transforms.Compose를 사용합니다.
    transform = transforms.Compose(
        [
            # 1. 이미지 크기 조절: 모든 이미지의 크기를 224x224 픽셀로 맞춥니다.
            transforms.Resize((224, 224)),  # 자르는 게 아니라 축소/확대
            #  🆘 현재 모든 영상의 크기가 1290 x 1080으로 동일하기 때문에 Resize를 해줘야하는지 모르겠습니다.
            # 2. 텐서(Tensor)로 변환: 이미지를 딥러닝 모델이 계산할 수 있는 숫자 행렬(텐서)로 바꿉니다.
            transforms.ToTensor(),  #
            # 3. 정규화(Normalize): 이미지의 픽셀 값 범위를 조정하여 모델이 더 빠르고 안정적으로 학습하도록 돕습니다.
            #    아래 mean과 std 값은 ImageNet 데이터셋에서 미리 계산된 값으로, 보통 그대로 많이 사용합니다.
            # RGB 에 대한 범위 값
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # --------------------------------
    # 이미지에 대한 경로, 일괄적으로 적용할 전처리 이 2가지를 인수로 넣어줌.
    # dataset = CustomData(video_path, transform)
    
    return DataLoader(
        dataset=CustomData(video_path, transform),
        batch_size=8,  # 컴퓨터사양에 따라서 메모리가 처리할 수 있는 양을 정해줘야합니다.
        # 2n승 으로 늘려주는 편이다. 2,4,8,16,32,64,128,~~256,
        # 정해주고, 메모리가 감당이 가능한지 확인해봐야합니다.
        shuffle=True,
        drop_last=False,  # 마지막에 남는 데이터도 사용합니다. (False: 버리지 않음)
    )

if __name__ == "__main__":
    dl = get_dataloader()
    print(dl)  # DataLoader 객체 정보 출력

