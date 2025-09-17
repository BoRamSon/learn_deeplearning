from torch.utils.data import Dataset  # 커스텀 데이터셋을 만들기 위함
import os
import cv2
import numpy as np
import torch
import random
from typing import List, Tuple


class CustomData(Dataset):
    """
    루트 경로(root) 아래 클래스별 폴더 구조를 스캔하고,
    클래스 단위로 70:30(기본) 비율로 train/valid를 분할하여 사용하는 커스텀 데이터셋.

    예시 디렉토리 구조:
        root/
            bump/
                abc.avi
                    def.mp4
                        fall-down/
            . . .
        ... (총 6개 클래스 폴더)
    """

    # ===========================================================================
    def __init__(
        self,
        root: str,  # ❌ 이거는 list가 아닙니다. 여기에는 단순 string root 경로만 들어옵니다.
                    # ❌ 그렇다면... 나의 가공된 영상 path list는 어디 가는거지??
        transform,
        split: str = "train",  # "train" 또는 "valid" (기본적으로 train)
        train_ratio: float = 0.7,  # train/valid 분할 비율 (기본적으로 70:30)
        seed: int = 42,  # 셔플 시드 (기본적으로 42)
    ):
        """
            # 위 매개변수들을 다른 곳에서 사용할 때의 예시

                # CustomData 인스턴스 생성 (train, valid 각각 생성)
                train_ds = CustomData(root=root, transform=transform, split="train", train_ratio=0.7, seed=42)
                valid_ds = CustomData(root=root, transform=transform, split="valid", train_ratio=0.7, seed=42)

                # DataLoader 인스턴스 생성 (train, valid 각각 생성)
                train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, drop_last=False)
                valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=False, num_workers=2, drop_last=False)
        """

        # 입력 인자 저장
        self.root = root
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed

        # 고정 시퀀스 길이(예: 16 프레임)
            # - 시퀀스 길이가 샘플마다 다르면 DataLoader가 배치를 만들 때 텐서 크기가 맞지 않아 오류가 납니다.
            # - 따라서 '자르거나(샘플링) / 늘리거나(패딩)' 해서 길이를 동일하게 맞춥니다.
        self.fixed_len = 16  # DataLoader 배칭을 위해 프레임 길이를 고정합니다.

        # 프로젝트에서 사용하는 클래스 목록(폴더명과 동일해야 함)
        self.class_names: List[str] = [
            "bump",
            "fall-down",
            "fall-off",
            "hit",
            "jam",
            "no-accident",
        ]

        # 지원하는 비디오 확장자
        self.allowed_exts = (".avi", ".mp4", ".mov", ".mkv", ".wmv")

        # --------------------------------------------------
        # 1) 루트 아래 클래스 폴더를 순회하여 클래스별 비디오 파일 목록 수집
        #   🔥 여기서 지금 클래스별로 영상 경로를 조회해서 클래스별 영상을 value로 가진 dictionary를 만들고 있습니다.
        #       즉 내가 가공했던 list가 아닌 여기서는 dictionary로 합니다....
        
        class_to_videos = {}

        for cname in self.class_names:
            cdir = os.path.join(self.root, cname)
            if not os.path.isdir(cdir):
                print(f"[경고] 클래스 폴더가 없습니다: {cdir}")
                class_to_videos[cname] = []
                continue
            vids = [
                os.path.join(cdir, fn)
                for fn in os.listdir(cdir)
                if fn.lower().endswith(self.allowed_exts)
            ]
            vids.sort()
            class_to_videos[cname] = vids

        # --------------------------------------------------
        # 2) 클래스별로 셔플 후 train/valid 분할(재현성 보장)
        #   🔥 여기서 영상을 train/valid로 분할하는 것이 아니라, 이 영상 path들을 train/valid로 분할하는 것입니다.

        rng = random.Random(self.seed)

        self.samples: List[Tuple[str, int]] = []  # train or valid의 비디오 경로(str), 라벨 인덱스(int)(클래스의 index입니다.)

        for idx, cname in enumerate(self.class_names):  # index 번호와 함께 클래스별로 1개씩 받습니다.
            vids = class_to_videos.get(cname, [])       # 클래스별로 영상 path를 list로 가져옵니다.
            if not vids:
                continue
            vids_copy = vids[:] # 굳이 전체를 모두 복사합니다.
            rng.shuffle(vids_copy)  # 복사된 리스트를 셔플합니다.
            n_total = len(vids_copy)    # 총 영상 수
            n_train = int(round(n_total * self.train_ratio))    # train/valid 분할 비율에 따라 train 영상 수
            train_list = vids_copy[:n_train]    # train 영상 path list (이런 식으로 분할을 하는 방법을 잘 익혀두세요.)
            valid_list = vids_copy[n_train:]    # valid 영상 path list (이런 식으로 분할을 하는 방법을 잘 익혀두세요.)

            if self.split == "train":   # split 인수가 train이라면,
                for vp in train_list:
                    self.samples.append((vp, idx))
            elif self.split == "valid": # split 인수가 valid라면,
                for vp in valid_list:
                    self.samples.append((vp, idx))
            else:
                raise ValueError("split은 'train' 또는 'valid'만 허용됩니다.")


    # ===========================================================================
    def __len__(self):
        # 전체 샘플 수 반환
        return len(self.samples)

    
    # ===========================================================================
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


    # ===========================================================================
    # 텐서 형태로 변환된 이미지를 돌려줌
    def __getitem__(self, idx):
        """
        목적:
            하나의 비디오를 모델 입력에 적합한 시퀀스 텐서로 변환합니다.
            - FPS를 30으로 맞춰 시간 해상도를 표준화하고,
            - 시퀀스 길이를 self.fixed_len(예: 16 프레임)으로 고정한 뒤,
            - 프레임별 전처리(Resize/Normalize 등)를 적용하여
            - (T, C, H, W) 형태의 텐서를 반환합니다.

        입력(내부적으로 사용되는 소스):
            - self.samples[idx]에서 (video_path, label_index)를 가져옵니다.
            - video_path는 실제 비디오 파일 경로, label_index는 클래스 인덱스입니다.

        처리 단계(요약):
            1) 비디오 로드 및 FPS 30으로 리샘플링
               - same_fps(video_path) -> numpy (T, H, W, C), BGR
            2) 예외 처리
               - 프레임이 없으면 더미(검은 화면) 프레임을 생성하여 길이를 self.fixed_len으로 채움
            3) 길이 고정
               - 길면 균등 샘플링으로 self.fixed_len개 선택
               - 짧으면 마지막 프레임 반복 패딩으로 길이 확장
            4) 프레임별 전처리
               - BGR -> RGB 변환 후, transform 적용
               - 권장 transform: ToPILImage -> Resize(224,224) -> ToTensor -> Normalize
            5) 시퀀스 텐서 생성
               - torch.stack으로 (T, C, H, W) 텐서 구성

        출력(반환 값):
            - result: torch.FloatTensor, shape=(T, C, H, W)
                예: T=16, C=3, H=224, W=224
            - label : int (클래스 인덱스)

        주의/가정:
            - OpenCV는 BGR 채널을 사용하므로 반드시 RGB로 변환해야 ImageNet 정규화 등과 호환됩니다.
            - transform이 numpy 입력을 받을 경우, ToPILImage가 맨 앞에 있어야 Resize가 정상 동작합니다.
            - 길이 고정은 배치 텐서 생성(B, T, C, H, W)에 필수입니다. 길이가 다르면 DataLoader가 에러를 냅니다.
            - self.fixed_len, 입력 해상도(Resize 크기)는 모델/메모리 상황에 따라 조정 가능.

        예시 사용:
            x, y = dataset[i]
            # x.shape == (T, C, H, W), y == int
        """
        # (비디오 경로, 라벨 인덱스) - 튜플로 된 리스트를 인덱스로 하나의 영상에 접근하여 가져옵니다.
        video, label = self.samples[idx]
        # 라벨명(디버깅용)
        human_accident_class_name = self.class_names[label]

        # -------------------------------------------------------------
        # same_fps는 “시간 간격”만 맞춰줍니다.
        # 🟢 🔥🔥🔥🔥🔥 “비디오에서 읽은 프레임 시퀀스를 예외 처리 → 길이 고정(샘플링/패딩) → 프레임별 전처리(BGR→RGB, Resize/Normalize) → (T, C, H, W) 텐서로 변환”하는 단계
        # 1) 비디오에서 프레임 시퀀스를 로드하고 FPS를 30으로 리샘플링합니다.
        frames = self.same_fps(video)  # 반환: numpy, shape=(T, H, W, C), 채널 순서=BGR(OpenCV 기본)

        # 2) 예외 처리: 비디오를 열지 못했거나 프레임이 비어있다면, 학습이 끊기지 않도록 더미 프레임을 생성합니다.
        if frames is None or len(frames) == 0:
            # 더미 프레임은 검은 화면(모두 0)으로 구성합니다. 길이는 self.fixed_len으로 맞춥니다.
            # 해상도는 224x224(모델 입력 크기에 보통 맞춤), 채널은 3(RGB 가정)이 됩니다.
            frames = np.zeros((self.fixed_len, 224, 224, 3), dtype=np.uint8)

        # 3) 시퀀스 길이를 고정(self.fixed_len)하기 위한 준비: 현재 길이 T를 구합니다.
        T = len(frames)  # 현재 시퀀스의 실제 프레임 개수

        # 4) 길이 고정 로직
        if T >= self.fixed_len:
            # 4-1) 프레임이 충분히 많거나 같으면, 균등 간격으로 self.fixed_len개를 골라
            #      전체 구간의 정보를 고르게 유지하면서 길이를 줄입니다.
            #      예: T=100, fixed_len=16 -> 0~99 구간에서 16개를 골라 다운샘플링
            indices = np.linspace(0, T - 1, self.fixed_len).astype(int)
            frames = frames[indices]
        else:
            # 4-2) 프레임이 부족하면, 마지막 프레임을 반복해서 붙이는 방식으로 길이를 늘립니다.
            #      이렇게 하면 시퀀스의 마지막 순간이 정지된 형태로 유지되어 모델 입력 차원을 맞출 수 있습니다.
            pad_count = self.fixed_len - T  # 부족한 개수만큼
            pad = np.repeat(frames[-1:], pad_count, axis=0)  # 마지막 프레임을 pad_count만큼 반복
            frames = np.concatenate([frames, pad], axis=0)  # 원래 프레임 뒤에 붙여 최종 길이 고정

        # 5) 전처리된 프레임 텐서를 쌓을 리스트를 준비합니다.
        processed = []  # 이 리스트에 (C,H,W) 텐서들을 순서대로 append합니다.

        # 6) 프레임을 한 장씩 꺼내어 전처리를 적용합니다.
        for f in frames:
            # 6-1) OpenCV는 BGR 채널 순서를 사용하므로, 일반적인 학습 파이프라인(RGB)에 맞추기 위해 변환합니다.
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)  # BGR -> RGB, shape 유지: (H,W,C)

            if self.transform is not None:
                # 6-2) transform 파이프라인 적용(권장 구성):
                #   ToPILImage() -> Resize(224,224) -> ToTensor() -> Normalize(...)
                # 주의: 입력이 numpy이면 ToPILImage가 먼저 있어야 Resize가 정상 작동합니다.
                img = self.transform(rgb)  # 기대 결과: torch.Tensor, shape=(C,H,W), dtype=float
            else:
                # 6-3) transform이 없다면 최소한의 변환을 직접 수행합니다.
                #      numpy(H,W,C)[0..255] -> tensor(C,H,W)[0..1]
                img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

            # 6-4) 처리된 프레임을 리스트에 추가합니다.
            processed.append(img)

        # 7) 리스트에 쌓인 프레임 텐서들을 시간축(T) 기준으로 스택하여 하나의 시퀀스 텐서를 만듭니다.
        #    최종 shape: (T, C, H, W)
        #    이후 DataLoader가 배치를 만들면 모델 입력은 (B, T, C, H, W)이 됩니다.
        result = torch.stack(processed, dim=0)
        # =================================================================================

        # 현재 영상에 대한 class를 index(번호)로 가져가 label로 붙여줍니다.
        return result, label