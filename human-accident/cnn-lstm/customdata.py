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
        video = cv2.VideoCapture(path)
        if not video.isOpened():
            return print("Could not Open :", path)

        target_fps = 30

        original_fps = int(video.get(cv2.CAP_PROP_FPS))
        print(f"현재 초당 프레임 수 : {original_fps}")

        frame_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"현재 총 프레임 수 : {frame_length}")

        frames = []
        count = 0

        while True:
            ret, img = video.read()  # 한 프레임 읽기
            if not ret:  # 더 이상 프레임 없으면 종료
                break

            if original_fps > target_fps:
                step = round(original_fps / target_fps)
                if count % step == 0:
                    frames.append(img)
            elif original_fps < target_fps:
                frames.append(img)
                if (count + 1) % original_fps == 0:
                    frames.append(img)
            else:
                frames.append(img)

            count += 1  # 읽은 프레임 수 증가

        video.release()

        return np.array(frames)

    # 텐서 형태로 변환된 이미지를 돌려줌
    def __getitem__(self, idx):
        human_accident_class_list = [
            "bump",
            "fall-down",
            "fall-off",
            "hit",
            "jam",
            "no-accident",
        ]  # class에 대한 list를 정의하였습니다.

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