# train_valid_dataset.py로 이동 및 수정

from torch.utils.data import Dataset  # 커스텀 데이터셋을 만들기 위함
import os
import cv2
import numpy as np


class CustomData(Dataset):
    def __init__(self, path, transform):
        self.video_path = path
        self.transform = transform

    def __len__(self):
        # if (이미지수 == 라벨수) return 이미지 수
        return len(self.video_path)
        # __len__은 DataLoader가 전체 데이터 개수를 알기 위해 호출합니다.

    def same_fps(self, path):  # 하나의 동영상 경로만 들어가서 처리합니다.
        """
        입력 비디오의 FPS를 기준 FPS(여기서는 30fps)에 맞춰서 프레임 시퀀스를 리샘플링합니다.
        - original_fps > 30  : 프레임을 건너뛰면서 다운샘플링
        - original_fps < 30  : 일부 프레임을 복제하여 업샘플링
        - original_fps == 30 : 그대로 사용

        반환 형태: numpy.ndarray, shape = (T, H, W, C), BGR(Channel 마지막)
        주의: OpenCV(cv2)는 기본 BGR 순서입니다. 학습 전처리에서는 RGB로 변환해야 합니다.
        """
        video = cv2.VideoCapture(path)
        if not video.isOpened():
            print("Could not Open :", path)
            return None

        target_fps = 30

        original_fps = int(video.get(cv2.CAP_PROP_FPS))
        # 디버깅용 로그: 원본 FPS
        print(f"현재 초당 프레임 수 : {original_fps}")

        frame_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # 디버깅용 로그: 총 프레임 수
        print(f"현재 총 프레임 수 : {frame_length}")

        frames = []
        count = 0

        while True:
            ret, img = video.read()  # 한 프레임 읽기
            if not ret:  # 더 이상 프레임 없으면 종료
                break

            if original_fps > target_fps:
                # 원본 FPS가 높으면 일정 간격(step)으로 프레임을 선택하여 다운샘플링
                step = round(original_fps / target_fps)
                if count % step == 0:
                    frames.append(img)
            elif original_fps < target_fps:
                # 원본 FPS가 낮으면 현재 프레임을 추가하고,
                # 한 초가 지날 때마다(원본 fps마다) 마지막 프레임을 한 번 더 추가하여 업샘플링
                frames.append(img)
                if (count + 1) % original_fps == 0:
                    frames.append(img)
            else:
                frames.append(img)

            count += 1  # 읽은 프레임 수 증가

        video.release()

        # numpy 배열로 변환: (T, H, W, C), BGR
        return np.array(frames)

    # 텐서 형태로 변환된 이미지를 돌려줌
    def __getitem__(self, idx):
        """
        하나의 비디오 샘플을 로드하여 다음을 수행합니다.
        1) 클래스 라벨 추출 (상위 폴더명 기준)
        2) 동일 FPS(30fps)로 리샘플링된 프레임 시퀀스 로드 (BGR)
        3) fixed_len(예: 16) 프레임 길이에 맞게 '균등 샘플링' 또는 '마지막 프레임 패딩'
        4) 각 프레임에 대해 BGR -> RGB 변환 후, transform(ToPILImage/Resize/ToTensor/Normalize) 적용
        5) (T, C, H, W) 텐서로 스택하여 반환

        반환:
            result: torch.FloatTensor, (T, C, H, W)
            label : int (클래스 인덱스)
        """
        human_accident_class_list = [
            "bump",
            "fall-down",
            "fall-off",
            "hit",
            "jam",
            "no-accident",
        ]  # class에 대한 list를 정의하였습니다.

        # 하나의 영상 파일 경로 선택
        video = self.video_path[idx]  # 하나의 영상 경로를 얻음

        # dirname = 마지막 파일 이름을 제거하고 상위 폴더 경로만 남긴 뒤 / basename = 맨 마지막 부분(폴더명 또는 파일명) 만 뽑아냅니다.
        human_accident_class_name = os.path.basename(os.path.dirname(video))

        video = self.same_fps(
            video
        )  # 하나의 영상 fps를 30프레임에 맞춰서 넘파이 배열 반환

        if self.transform is not None:
            result = self.transform(video)

        # 현재 영상에 대한 class를 index(번호)로 가져가 label로 붙여줍니다.
        label = human_accident_class_list.index(human_accident_class_name)

        return result, label