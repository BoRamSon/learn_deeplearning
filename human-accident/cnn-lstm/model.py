import torch  # PyTorch 라이브러리 임포트
from torch import nn  # PyTorch의 신경망 모듈(nn) 임포트

from models import cnnlstm  # models 폴더의 cnnlstm.py 파일에서 모델 클래스를 임포트

def generate_model(opt, device):  # 모델을 생성하고 지정된 장치로 보내는 함수
	# opt: 커맨드라인 인자를 담고 있는 객체
	# device: 모델을 올릴 장치 (e.g., 'cuda:0' 또는 'cpu')
	assert opt.model in [  # opt.model 값이 'cnnlstm'인지 확인. 다른 값이면 에러 발생
		'cnnlstm'  # 현재 지원하는 모델 리스트
	]  # assert 문은 조건이 참이 아니면 AssertionError를 발생시켜 프로그램을 중단시킴

	if opt.model == 'cnnlstm':  # opt.model 값이 'cnnlstm'인 경우
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)  # cnnlstm.py의 CNNLSTM 클래스를 인스턴스화. 출력 클래스 개수를 opt.n_classes로 설정.
	return model.to(device)  # 생성된 모델을 지정된 device(GPU 또는 CPU)로 이동시킨 후 반환