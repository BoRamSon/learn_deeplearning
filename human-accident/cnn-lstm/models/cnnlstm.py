import torch            # PyTorch 라이브러리 임포트
import torch.nn as nn   # PyTorch의 신경망 모듈(nn) 임포트
import torchvision.models as models  # torchvision에서 제공하는 사전 학습된 모델들을 임포트
from torch.nn.utils.rnn import pack_padded_sequence  # 패딩된 시퀀스를 압축하기 위한 유틸리티 (현재 코드에서는 사용되지 않음)
import torch.nn.functional as F  # 활성화 함수(ReLU 등)와 같은 함수형 API를 포함하는 모듈
from torchvision.models import resnet101  # torchvision에서 ResNet-101 모델을 직접 임포트


class CNNLSTM(nn.Module):  # nn.Module을 상속받아 CNN-LSTM 모델 클래스를 정의
    def __init__(self, num_classes=2):  # 모델 초기화 메서드. num_classes는 최종 분류할 클래스의 수
        super(CNNLSTM, self).__init__()  # 부모 클래스(nn.Module)의 초기화 메서드 호출
        # CNN 특징 추출기(Feature Extractor) 정의
        self.resnet = resnet101(pretrained=True)  # ImageNet으로 사전 학습된 ResNet-101 모델을 로드
        # ResNet의 마지막 Fully Connected Layer를 새로운 레이어로 교체
        # 기존 ResNet-101의 출력 차원을 300으로 변경하여 LSTM의 입력으로 사용
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        
        # LSTM 시퀀스 모델 정의
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)  # 입력 크기 300, 은닉 상태 크기 256, 3개의 레이어로 구성된 LSTM
        
        # 분류기(Classifier) 정의
        self.fc1 = nn.Linear(256, 128)  # LSTM의 출력(256)을 받아 128 차원으로 변환하는 Fully Connected Layer
        self.fc2 = nn.Linear(128, num_classes)  # 128 차원을 최종 클래스 수(num_classes)로 변환하는 출력 레이어

    def forward(self, x_3d):  # 모델의 순전파 로직 정의
        # 입력 x_3d의 차원: (batch_size, sequence_length, C, H, W)
        hidden = None  # LSTM의 초기 은닉 상태(hidden state)와 셀 상태(cell state)를 None으로 초기화 (자동으로 0으로 채워짐)
        
        # 비디오의 각 프레임(시퀀스)을 순회
        for t in range(x_3d.size(1)):  # t는 0부터 sequence_length-1 까지의 프레임 인덱스
            # with torch.no_grad() 블록: CNN 부분의 가중치가 학습되지 않도록(gradient 계산을 멈추도록) 설정. 사전 학습된 가중치를 고정(freeze)하는 효과.
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # 현재 t번째 프레임(batch_size, C, H, W)을 ResNet에 통과시켜 특징 벡터를 추출
            
            # LSTM에 프레임 특징 벡터를 입력
            out, hidden = self.lstm(x.unsqueeze(0), hidden)  # 추출된 특징 벡터 x를 (1, batch_size, 300) 형태로 변환하여 LSTM에 입력. 이전 hidden 상태를 사용.

        # 분류기를 통해 최종 예측 수행
        x = self.fc1(out[-1, :, :])  # LSTM의 마지막 시점(time step)의 출력(out[-1])을 fc1 레이어에 통과시킴
        x = F.relu(x)  # ReLU 활성화 함수 적용
        x = self.fc2(x)  # 최종 출력 레이어(fc2)를 통과시켜 클래스별 점수(logit)를 계산
        return x  # 최종 예측 결과 반환