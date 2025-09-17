import torch  # PyTorch 라이브러리 임포트
import torch.nn as nn  # 신경망 모듈(레이어, 손실 함수 등)을 포함하는 PyTorch 라이브러리
import torch.optim as optim  # 최적화 알고리즘(Adam, SGD 등)을 포함하는 PyTorch 라이브러리
import tensorboardX  # TensorBoard 시각화를 위한 라이브러리
import os  # 운영체제와 상호작용하기 위한 라이브러리 (파일 경로, 디렉토리 생성 등)
import random  # 난수 생성을 위한 라이브러리
import numpy as np  # 수치 연산을 위한 라이브러리

from torch.utils.data import DataLoader  # 데이터셋을 미니배치로 나누어 로드하는 PyTorch 유틸리티
from torch.optim import lr_scheduler  # 학습률 스케줄러를 위한 PyTorch 유틸리티

# -------------------------------------------------------------------

from opts import parse_opts  # 커맨드라인 인자 파싱을 위한 opts.py에서 parse_opts 함수 임포트

from train import train_epoch  # 학습 로직이 구현된 train.py에서 train_epoch 함수 임포트

from validation import val_epoch  # 검증 로직이 구현된 validation.py에서 val_epoch 함수 임포트

from model import generate_model  # 모델 아키텍처를 생성하는 model.py에서 generate_model 함수 임포트

from dataset import get_training_set, get_validation_set  # 데이터셋을 가져오는 dataset.py의 함수들 임포트

from mean import get_mean, get_std  # 데이터 정규화를 위한 평균, 표준편차 값을 가져오는 mean.py의 함수들 임포트


from spatial_transforms import (    # 이미지 프레임에 적용할 공간적 변환(augmentation) 함수들 임포트
    Compose,                        # 여러 변환을 묶어주는 클래스
    Normalize,                      # 텐서를 평균과 표준편차로 정규화하는 클래스
    Scale,                          # 이미지 크기를 조절하는 클래스
    CenterCrop,                     # 이미지 중앙을 잘라내는 클래스
    CornerCrop,                     # 이미지 모서리를 잘라내는 클래스
    MultiScaleCornerCrop,           # 여러 스케일로 모서리를 잘라내는 클래스
    MultiScaleRandomCrop,           # 여러 스케일로 무작위로 잘라내는 클래스
    RandomHorizontalFlip,           # 이미지를 무작위로 좌우 반전하는 클래스
    ToTensor,                       # PIL 이미지나 'numpy 배열'을 '텐서'로 변환하는 클래스
)

from temporal_transforms import LoopPadding, TemporalRandomCrop  # 비디오 시퀀스에 적용할 시간적 변환 함수들 임포트

from target_transforms import ClassLabel, VideoID  # 타겟(라벨) 데이터에 적용할 변환 함수들 임포트

from target_transforms import Compose as TargetCompose  # 타겟 변환을 묶어주기 위한 Compose 클래스 임포트


def resume_model(opt, model, optimizer):
    """
    Resume model

    체크포인트(checkpoint)로부터 모델과 옵티마이저 상태를 불러와 학습을 이어갈 준비를 하는 함수.
    입력:
        opt       : argparse로 파싱된 옵션 객체 (opt.resume_path 가 체크포인트 경로여야 함)
        model     : 미리 생성된 PyTorch 모델 인스턴스
        optimizer : 미리 생성된 PyTorch 옵티마이저 인스턴스
    반환:
        start_epoch : 다음에 학습을 시작할 epoch 번호 (보통 저장된 epoch + 1)
    주의:
        - 체크포인트 파일의 내부 구조는 { "epoch": int, "state_dict": model_state_dict,
                                    "optimizer_state_dict": optimizer_state_dict } 
                                형태를 기대함.
    """
    # 1) 디스크에서 체크포인트 파일을 로드
    # torch.load는 파일에 저장된 파이썬 객체(여기서는 dict)를 복원함.
    # 주의: GPU에서 저장된 체크포인트를 CPU 환경에서 불러오면 오류가 날 수 있으므로
    #       실제 코드에서는 map_location=device 사용을 권장.
    checkpoint = torch.load(opt.resume_path)

    # 2) 체크포인트 안의 모델 파라미터(state_dict)를 현재 모델 객체에 적용
    # checkpoint["state_dict"]는 파라미터 이름->텐서 매핑 딕셔너리여야 함.
    # 만약 저장할 때 DataParallel(model)을 사용해 저장했다면 키 앞에 'module.'이 붙어있을 수 있다.
    # 모델 구조가 현재 모델과 완전히 일치하지 않으면 load_state_dict에서 에러가 발생할 수 있음.
    model.load_state_dict(checkpoint["state_dict"])

    # 3) 체크포인트 안의 옵티마이저 상태(state_dict)를 현재 옵티마이저에 적용
    # 옵티마이저 상태에는 learning rate, 모멘텀 버퍼, param_groups, 그리고 내부 모멘텀 추정값 등이 포함됨.
    # 옵티마이저 구조(param_groups 등)가 다르면 load_state_dict에서 예외가 발생할 수 있음.
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 4) 복원된 체크포인트가 언제 저장된 것인지 사용자에게 출력
    # checkpoint["epoch"]는 일반적으로 체크포인트를 저장한 에폭 번호(정수)를 가짐.
    print("Model Restored from Epoch {}".format(checkpoint["epoch"]))

    # 5) 함수 호출자에게 반환할 '다음 시작 에폭' 결정
    # 보통 저장된 epoch가 5라면, 우리는 6번째 에폭(epoch=6)부터 학습을 이어가므로 +1 해준다.
    start_epoch = checkpoint["epoch"] + 1

    # 6) 계산된 시작 에폭 반환
    return start_epoch



def get_loaders(opt):
    """Make dataloaders for train and validation sets"""
    # ----------------------------------
    # ✅ train loader
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)  # 정규화에 사용할 평균값 계산
    if opt.no_mean_norm and not opt.std_norm:  # 평균과 표준편차 정규화를 모두 사용하지 않는 경우
        norm_method = Normalize([0, 0, 0], [1, 1, 1])  # 정규화를 수행하지 않는 것과 같은 효과 (0을 빼고 1로 나눔)
    elif not opt.std_norm:  # 평균 정규화만 사용하는 경우
        norm_method = Normalize(opt.mean, [1, 1, 1])  # 평균값만 빼고, 1로 나누어 표준편차 정규화는 생략
    else:  # 평균과 표준편차 정규화를 모두 사용하는 경우
        norm_method = Normalize(opt.mean, opt.std)  # 평균을 빼고 표준편차로 나눔
    
    spatial_transform = Compose(  # 학습 데이터에 적용할 공간적 변환들을 정의
        [
            # crop_method,
            Scale((opt.sample_size, opt.sample_size)),  # 이미지 크기를 (sample_size, sample_size)로 조절
            # RandomHorizontalFlip(),
            ToTensor(opt.norm_value),  # 이미지를 텐서로 변환하고 norm_value로 나눔
            norm_method,  # 위에서 정의한 정규화 방법 적용
        ]
    )
    
    temporal_transform = TemporalRandomCrop(16)  # 비디오 프레임 시퀀스에서 16 프레임을 무작위로 잘라냄
    
    target_transform = ClassLabel()  # 타겟 데이터에서 클래스 라벨만 추출
    
    training_data = get_training_set(  # 학습 데이터셋 객체 생성
        opt, spatial_transform, temporal_transform, target_transform  # 옵션과 변환들을 전달
    )
    
    train_loader = torch.utils.data.DataLoader(  # 학습 데이터 로더 생성
        training_data,  # 위에서 생성한 데이터셋 객체
        batch_size=opt.batch_size,  # 배치 크기 설정
        shuffle=True,  # 에폭마다 데이터를 섞을지 여부 (학습 시에는 True가 일반적)
        num_workers=opt.num_workers,  # 데이터 로딩에 사용할 서브프로세스 수
        pin_memory=True,  # GPU로 데이터를 더 빨리 전송하기 위해 메모리에 고정
    )

    # ----------------------------------
    # ✅ validation loader
    spatial_transform = Compose(  # 검증 데이터에 적용할 공간적 변환들을 정의 (보통 augmentation 제외)
        [
            Scale((opt.sample_size, opt.sample_size)),  # 이미지 크기를 (sample_size, sample_size)로 조절
            # CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value),  # 이미지를 텐서로 변환하고 norm_value로 나눔
            norm_method,  # 학습 데이터와 동일한 정규화 방법 적용
        ]
    )
    target_transform = ClassLabel()  # 타겟 데이터에서 클래스 라벨만 추출
    temporal_transform = LoopPadding(16)  # 비디오 프레임 시퀀스가 16보다 짧을 경우, 앞부분을 반복해서 16 프레임으로 만듦
    validation_data = get_validation_set(  # 검증 데이터셋 객체 생성
        opt, spatial_transform, temporal_transform, target_transform  # 옵션과 변환들을 전달
    )
    val_loader = torch.utils.data.DataLoader(  # 검증 데이터 로더 생성
        validation_data,  # 위에서 생성한 데이터셋 객체
        batch_size=opt.batch_size,  # 배치 크기 설정
        shuffle=False,  # 검증 시에는 데이터를 섞지 않음 (일관된 평가를 위해)
        num_workers=opt.num_workers,  # 데이터 로딩에 사용할 서브프로세스 수
        pin_memory=True,  # GPU로 데이터를 더 빨리 전송하기 위해 메모리에 고정
    )
    return train_loader, val_loader  # 생성된 학습 및 검증 데이터 로더 반환

# --------------------------------------------------------------------
# 2. main_worker() 함수를 들여다보자~
def main_worker():
    # 3. parsing
    opt = parse_opts()  # 커맨드라인 인자들을 파싱하여 opt 객체에 저장
    # parser : compiler의 일부로 컴파일러나 인터프리터에서 원시 프로그램을 읽어 들여 그 문장의 구조를 알아내는 parsing(구문 분석)을 행하는 프로그램
    # opt = options(옵션들) 의 줄임말 / 커맨드라인에서 받은 설정값들을 모아둔 객체
    print(opt)  # 파싱된 옵션들을 출력하여 확인

    # 🚨 4. 재현성을 위해 시드(seed) 고정
    seed = 1  # 시드 값 설정
    random.seed(seed)  # 파이썬 내장 random 모듈의 시드 고정
    np.random.seed(seed)  # numpy의 시드 고정
    torch.manual_seed(seed)  # PyTorch의 CPU 연산에 대한 시드 고정
    # 이렇게 고정하는 이유 : 내가 만든 모델을 다시 돌렸을 때, 혹은 다른 사람이 내 코드를 돌렸을 때 결과가 똑같이 나와야 합니다. 그래야 실험 결과를 신뢰할 수 있음.

    # 5. CUDA for PyTorch
    device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")  # --use_cuda 옵션이 있으면 지정된 GPU를, 없으면 CPU를 사용하도록 설정

    # 6. tensorboard
    summary_writer = tensorboardX.SummaryWriter(log_dir="tf_logs")  # TensorBoard 로그를 저장할 디렉토리 설정 및 writer 객체 생성
    """ 
    🟡 SummaryWriter   
        ㄴ 로그(스칼라, 이미지, 히스토그램, 그래프 등)를 디스크에 기록하는 "쓰기 도구(writer)" 클래스입니다.
        ㄴ 이 객체의 add_s calar, add_image, add_histogram, add_graph 등 메서드를 통해 이벤트를 기록합니다. 
    🟡 log_dir="tf_logs"  ->   이벤트 파일을 저장할 디렉토리 경로
    """
    
    # 7. defining model
    model = generate_model(opt, device)  # 설정(opt)에 맞는 model(model.py 파일 참조)을 생성하고 지정된 장치(device)로 이동

    # 8. get data loaders
    train_loader, val_loader = get_loaders(opt)  # 학습 및 검증 데이터 로더 생성

    # 9. optimizer - 신경망의 가중치(파라미터)를 학습 데이터에 맞게 업데이트해주는 알고리즘
    crnn_params = list(model.parameters())  # 최적화할 모델의 파라미터들을 리스트로 가져옴
    optimizer = torch.optim.Adam(  # Adam 옵티마이저 생성
        crnn_params, lr=opt.lr_rate, weight_decay=opt.weight_decay  # 모델 파라미터, 학습률, 가중치 감쇠(L2 정규화) 설정
    )
    # 🔎 optimizer 설명
        # 1. 신경망은 처음에 랜덤한 가중치(weight) 로 시작합니다.
        # 2. 입력 데이터를 통과시켜 출력(prediction) 을 계산합니다.
        # 3. 출력과 정답(label)을 비교해 손실(loss) 을 계산합니다.
        # 4. 역전파(backpropagation) 로 각 가중치가 손실에 얼마나 영향을 미쳤는지 기울기(gradient) 를 구합니다.
        # 5. 마지막으로 optimizer 가 그 기울기를 사용해서 가중치를 조금씩 조정합니다.
        # 이 과정을 여러 번 반복하면서 모델이 점점 "데이터에 잘 맞게" 학습됩니다.

    
    # 10. 학습률 스케줄러(Learning Rate Scheduler) 설정 코드
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    # 	optimizer, 'min', patience=opt.lr_patience 
    # )

    # 11. 손실함수
    criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류를 위한 CrossEntropy 손실 함수 정의

    # 12. resume model - 어디서 학습을 시작할 것인가?
    if opt.resume_path:  # “truthy”(빈 문자열/None/False가 아닌 값)인지 검사 / --resume_path 옵션으로 체크포인트 경로가 주어진 경우
        start_epoch = resume_model(opt, model, optimizer)  # 모델과 옵티마이저 상태를 복원하고 시작 에폭을 받아옴
    else:  # 체크포인트 경로가 주어지지 않은 경우 (처음부터 학습)
        start_epoch = 1  # 1 에폭부터 학습 시작

    # 13. start training - 학습 시작
    for epoch in range(start_epoch, opt.n_epochs + 1):  # 시작 에폭부터 마지막 에폭까지 반복
        train_loss, train_acc = train_epoch(  # 1 에폭 동안 모델을 학습
            model, train_loader, criterion, optimizer, epoch, opt.log_interval, device  # 필요한 모든 객체와 설정을 전달
        )
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)  # 1 에폭 학습 후 검증 데이터로 모델 성능 평가

        # saving weights to checkpoint
        if (epoch) % opt.save_interval == 0:  # 현재 에폭이 'save_interval'로 나누어 떨어질 때마다
            # scheduler.step(val_loss)
            # write summary
            summary_writer.add_scalar(  # TensorBoard에 학습 손실 기록
                "losses/train_loss", train_loss, global_step=epoch  # 'losses/train_loss' 태그로, 현재 에폭을 x축으로 하여 기록
            )
            summary_writer.add_scalar("losses/val_loss", val_loss, global_step=epoch)  # TensorBoard에 검증 손실 기록
            summary_writer.add_scalar(  # TensorBoard에 학습 정확도 기록
                "acc/train_acc", train_acc * 100, global_step=epoch  # 백분율로 변환하여 기록
            )
            summary_writer.add_scalar("acc/val_acc", val_acc * 100, global_step=epoch)  # TensorBoard에 검증 정확도 기록

            state = {  # 체크포인트로 저장할 정보들을 딕셔너리로 구성
                "epoch": epoch,  # 현재 에폭 번호
                "state_dict": model.state_dict(),  # 모델의 가중치
                "optimizer_state_dict": optimizer.state_dict(),  # 옵티마이저의 상태
            }
            torch.save(  # 체크포인트 파일 저장
                state,  # 저장할 딕셔너리
                os.path.join(  # 저장 경로와 파일 이름 생성
                    "snapshots", f"{opt.model}-Epoch-{epoch}-Loss-{val_loss:.4f}.pth"  # snapshots 폴더에 모델, 에폭, 손실 정보를 포함한 이름으로 저장
                ),
            )
            print("Epoch {} model saved!\n".format(epoch))  # 모델 저장 완료 메시지 출력


if __name__ == "__main__":
    # 1. main_worker 함수가 최초 시작지점입니다.
    main_worker()  # 스크립트가 직접 실행될 때 main_worker 함수 호출
