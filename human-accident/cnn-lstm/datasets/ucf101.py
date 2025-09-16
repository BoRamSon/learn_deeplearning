import torch  # PyTorch 라이브러리 임포트
import torch.utils.data as data  # PyTorch의 데이터 로딩 유틸리티 임포트
from PIL import Image  # 이미지 처리를 위한 Pillow 라이브러리 임포트
import os  # 운영체제와 상호작용하기 위한 라이브러리 (파일 경로 처리 등)
import math  # 수학 함수 사용을 위한 라이브러리
import functools  # 고차 함수와 호출 가능한 객체를 위한 라이브러리
import json  # JSON 파일 처리를 위한 라이브러리
import copy  # 객체 복사를 위한 라이브러리

from utils import load_value_file  # utils.py에서 텍스트 파일의 숫자 값을 읽어오는 함수 임포트


def pil_loader(path):  # Pillow를 사용하여 이미지를 로드하는 함수
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:  # 파일을 바이너리 읽기 모드로 열기
        with Image.open(f) as img:  # Pillow를 사용하여 이미지 파일 열기
            return img.convert('RGB')  # 이미지를 RGB 형식으로 변환하여 반환


def accimage_loader(path):  # accimage(Pillow보다 빠른 이미지 로더)를 사용하여 이미지를 로드하는 함수
    try:  # accimage 로드를 시도
        import accimage  # accimage 라이브러리 임포트
        return accimage.Image(path)  # accimage로 이미지 로드
    except IOError:  # accimage 로드 중 에러 발생 시
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)  # Pillow 로더를 대신 사용


def get_default_image_loader():  # 시스템에 설치된 백엔드에 따라 기본 이미지 로더를 반환하는 함수
    from torchvision import get_image_backend  # torchvision에서 현재 설정된 이미지 백엔드를 가져옴
    if get_image_backend() == 'accimage':  # 백엔드가 'accimage'인 경우
        return accimage_loader  # accimage 로더 반환
    else:  # 그렇지 않은 경우 (기본값은 'pillow')
        return pil_loader  # Pillow 로더 반환


def video_loader(video_dir_path, frame_indices, image_loader):  # 비디오 프레임들을 로드하는 함수
    video = []  # 프레임 이미지를 담을 리스트 초기화
    for i in frame_indices:  # 주어진 프레임 인덱스 목록을 순회
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))  # 프레임 이미지 파일 경로 생성 (e.g., image_00001.jpg)
        if os.path.exists(image_path):  # 해당 경로에 파일이 존재하면
            video.append(image_loader(image_path))  # 이미지 로더로 이미지를 로드하여 리스트에 추가
        else:  # 파일이 존재하지 않으면 (비디오의 끝 등)
            return video  # 현재까지 로드된 프레임 리스트를 반환

    return video  # 모든 프레임 로드 후 리스트 반환


def get_default_video_loader():  # 기본 비디오 로더를 생성하는 함수
    image_loader = get_default_image_loader()  # 기본 이미지 로더를 가져옴
    return functools.partial(video_loader, image_loader=image_loader)  # video_loader 함수의 image_loader 인자를 미리 채운 새로운 함수를 생성하여 반환


def load_annotation_data(data_file_path):  # JSON 어노테이션 파일을 로드하는 함수
    with open(data_file_path, 'r') as data_file:  # 어노테이션 파일을 읽기 모드로 열기
        return json.load(data_file)  # JSON 데이터를 파싱하여 파이썬 객체로 반환


def get_class_labels(data):  # 어노테이션 데이터에서 클래스 이름과 인덱스 매핑을 생성하는 함수
    class_labels_map = {}  # 클래스 이름: 인덱스 형태의 딕셔너리 초기화
    index = 0  # 인덱스 초기화
    for class_label in data['labels']:  # 어노테이션 데이터의 'labels' 리스트를 순회
        class_labels_map[class_label] = index  # 딕셔너리에 클래스 이름과 현재 인덱스를 추가
        index += 1  # 인덱스 증가
    return class_labels_map  # 생성된 매핑 딕셔너리 반환


def get_video_names_and_annotations(data, subset):  # 특정 서브셋(training/validation)에 해당하는 비디오 목록과 어노테이션을 가져오는 함수
    video_names = []  # 비디오 이름을 담을 리스트
    annotations = []  # 어노테이션을 담을 리스트

    for key, value in data['database'].items():  # 어노테이션의 'database' 딕셔너리를 순회
        this_subset = value['subset']  # 현재 비디오의 서브셋 정보
        if this_subset == subset:  # 현재 비디오의 서브셋이 원하는 서브셋과 일치하면
            label = value['annotations']['label']  # 비디오의 라벨(클래스 이름)을 가져옴
            video_names.append('{}/{}'.format(label, key))  # '클래스이름/비디오ID' 형태로 비디오 이름을 생성하여 리스트에 추가
            annotations.append(value['annotations'])  # 해당 비디오의 어노테이션 정보를 리스트에 추가

    return video_names, annotations  # 비디오 이름 리스트와 어노테이션 리스트를 반환


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):  # 데이터셋을 구성하는 메인 함수
    data = load_annotation_data(annotation_path)  # 어노테이션 파일 로드
    video_names, annotations = get_video_names_and_annotations(data, subset)  # 원하는 서브셋의 비디오 정보 가져오기
    class_to_idx = get_class_labels(data)  # 클래스 이름 -> 인덱스 매핑 생성
    idx_to_class = {}  # 인덱스 -> 클래스 이름 매핑 생성
    for name, label in class_to_idx.items():  # class_to_idx를 순회하며
        idx_to_class[label] = name  # 반대 방향의 매핑을 생성

    dataset = []  # 최종 데이터셋 샘플들을 담을 리스트
    for i in range(len(video_names)):  # 각 비디오에 대해 반복
        if i % 1000 == 0:  # 1000개 비디오마다 진행 상황 출력
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])  # 비디오 프레임들이 저장된 디렉토리 경로 생성
        if not os.path.exists(video_path):  # 해당 경로가 존재하지 않으면
            continue  # 다음 비디오로 넘어감

        n_frames_file_path = os.path.join(video_path, 'n_frames')  # 총 프레임 수가 저장된 파일 경로
        n_frames = int(load_value_file(n_frames_file_path))  # 파일에서 총 프레임 수를 읽어옴
        if n_frames <= 0:  # 프레임 수가 0 이하면
            continue  # 다음 비디오로 넘어감

        begin_t = 1  # 시작 프레임 인덱스
        end_t = n_frames  # 끝 프레임 인덱스
        sample = {  # 하나의 샘플 정보를 담을 딕셔너리 생성
            'video': video_path,  # 비디오 경로
            'segment': [begin_t, end_t],  # 비디오 전체 구간
            'n_frames': n_frames,  # 총 프레임 수
            'video_id': video_names[i].split('/')[1]  # 비디오 고유 ID
        }
        if len(annotations) != 0:  # 어노테이션 정보가 있으면
            sample['label'] = class_to_idx[annotations[i]['label']]  # 클래스 이름을 인덱스로 변환하여 저장
        else:  # 없으면 (e.g. 테스트셋)
            sample['label'] = -1  # 라벨을 -1로 설정

        if n_samples_for_each_video == 1:  # 비디오당 1개의 샘플만 사용하는 경우
            sample['frame_indices'] = list(range(1, n_frames + 1))  # 모든 프레임 인덱스를 리스트로 저장
            dataset.append(sample)  # 데이터셋에 샘플 추가
        else:  # 비디오당 여러 샘플을 사용하는 경우 (클립 단위로 자름)
            if n_samples_for_each_video > 1:  # 샘플 수가 1보다 크면
                step = max(1,  # 클립을 추출할 간격(step) 계산
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:  # 샘플 수가 1 이하이면 (일반적으로 이 경우는 없음)
                step = sample_duration  # 스텝을 샘플 길이로 설정
            for j in range(1, n_frames, step):  # 계산된 간격으로 프레임 인덱스를 순회
                sample_j = copy.deepcopy(sample)  # 기본 샘플 정보 복사
                sample_j['frame_indices'] = list(  # 현재 시작점(j)부터 일정 길이(sample_duration)만큼의 프레임 인덱스를 잘라 저장
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)  # 잘라낸 클립을 하나의 샘플로 데이터셋에 추가

    return dataset, idx_to_class  # 완성된 데이터셋 리스트와 인덱스->클래스 매핑 반환


class UCF101(data.Dataset):  # PyTorch의 Dataset 클래스를 상속받는 UCF101 데이터셋 클래스
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples (주석 설명, 실제 코드와는 다름)
    """

    def __init__(self,  # 클래스 초기화 메서드
                 root_path,  # 데이터셋 루트 경로
                 annotation_path,  # 어노테이션 파일 경로
                 subset,  # 'training' 또는 'validation'
                 n_samples_for_each_video=1,  # 비디오당 샘플 수
                 spatial_transform=None,  # 프레임에 적용할 공간적 변환
                 temporal_transform=None,  # 프레임 시퀀스에 적용할 시간적 변환
                 target_transform=None,  # 라벨에 적용할 변환
                 sample_duration=16,  # 클립의 길이 (프레임 수)
                 get_loader=get_default_video_loader):  # 사용할 비디오 로더 함수
        self.data, self.class_names = make_dataset(  # make_dataset 함수를 호출하여 데이터셋을 생성
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform  # 공간적 변환 저장
        self.temporal_transform = temporal_transform  # 시간적 변환 저장
        self.target_transform = target_transform  # 타겟 변환 저장
        self.loader = get_loader()  # 비디오 로더 함수 저장

    def __getitem__(self, index):  # 데이터셋에서 특정 인덱스의 샘플을 가져오는 메서드
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']  # 해당 인덱스의 비디오 경로 가져오기

        frame_indices = self.data[index]['frame_indices']  # 해당 인덱스의 프레임 인덱스 리스트 가져오기
        if self.temporal_transform is not None:  # 시간적 변환이 지정되어 있으면
            frame_indices = self.temporal_transform(frame_indices)  # 프레임 인덱스에 변환 적용 (e.g., TemporalRandomCrop)
        clip = self.loader(path, frame_indices)  # 비디오 로더를 사용해 프레임 이미지들을 로드
        if self.spatial_transform is not None:  # 공간적 변환이 지정되어 있으면
            self.spatial_transform.randomize_parameters()  # 변환 파라미터 랜덤화 (e.g., RandomCrop의 위치)
            clip = [self.spatial_transform(img) for img in clip]  # 각 프레임 이미지에 공간적 변환 적용
        clip = torch.stack(clip, 0)  # 프레임 텐서 리스트를 하나의 텐서로 합침 (차원: T x C x H x W)

        target = self.data[index]  # 해당 인덱스의 라벨 정보(딕셔너리) 가져오기
        if self.target_transform is not None:  # 타겟 변환이 지정되어 있으면
            target = self.target_transform(target)  # 라벨 정보에 변환 적용 (e.g., ClassLabel)
        return clip, target  # 최종적으로 변환된 클립 텐서와 타겟을 반환

    def __len__(self):  # 데이터셋의 총 샘플 수를 반환하는 메서드
        return len(self.data)  # self.data 리스트의 길이(총 클립 수)를 반환
