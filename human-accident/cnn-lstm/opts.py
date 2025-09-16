# 파싱(parsing) : 문자열 형태로 주어진 데이터를 쪼개고 분석해서, 우리가 쓸 수 있는 구조화된 데이터로 바꾸는 과정.

import argparse     # argparse 모듈 임포트: 커맨드라인 인자 파싱(우리가 터미널에서 프로그램을 실행할 때 뒤에 붙이는 옵션들을 말함)을 위한 표준 라이브러리.
                    # - argparse는 sys.argv 를 읽어서 정의한 인자 규격에 따라 값을 파싱,
                    #   타입 변환, 자동 help 생성, 에러 처리까지 제공한다.

def parse_opts():  # parse_opts 함수 정의: 프로그램(또는 스크립트)에서 사용할 모든 CLI 옵션을 정의하고 파싱하여 반환하는 역할.
                    # - 반환값은 argparse.Namespace 객체. 호출부에서 args.dataset 등으로 접근.
                    # - 의도: 재현 가능한 학습/추론 설정을 커맨드라인으로 주입하기 위함.
	parser = argparse.ArgumentParser()  # ArgumentParser 인스턴스 생성. 
                                        # argparse.ArgumentParser() 객체는 “명령어 옵션(인자)”들을 정의해야 합니다.
                                        # - 기본적으로 프로그램 이름을 자동으로 추출해 help에 표시.
                                        # - 필요하면 description, formatter_class 등을 인수로 넣어 더 친절한 help를 만들 수 있다.
    
    # ---------------------------------------------------------------------------------------------------
	# < 설명 >
    # parser.add_argument(
    #     "--batch_size",    # (1) 옵션 이름 (터미널에서 입력할 이름)
    #     type=int,          # (2) 자료형 변환 (문자열 → 정수)
    #     default=32,        # (3) 입력 안 했을 때 기본값
    #     help="배치 크기"   # (4) 사용법 설명 (도움말에 표시됨)
    # )
    # ---------------------------------------------------------------------------------------------------
	parser.add_argument('--dataset', type=str,  # '--dataset' 옵션 추가: 옵션 이름(하이픈 두 개는 optional argument).
						default='uf101', help='dataset type')  # default='uf101' (문자열), help는 --help 출력에 노출됨.
                                                                            # - 파싱 후 args.dataset 으로 접근.
                                                                            # - type=str 이므로 전달값이 문자열이 아니면 에러 발생.
                                                                            # - 사용 예: `--dataset ucf101`
	parser.add_argument(
		'--root_path',
		default='/root/data/ActivityNet',
		type=str,
		help='Root directory path of data')     # '--root_path': 데이터 루트 디렉토리 경로. 절대/상대경로 모두 가능.
                                                # - 기본값은 '/root/data/ActivityNet'로 설정되어 있으므로, 로컬 환경에 맞게 변경 필요.
                                                # - 파일/디렉토리 존재 여부는 argparse에서 검사하지 않으므로 코드에서 체크 권장.
	parser.add_argument(
		'--video_path',
		default='video_kinetics_jpg',
		type=str,
		help='Directory path of Videos')    # '--video_path': 비디오(혹은 프레임 이미지)들이 저장된 서브디렉토리 경로.
                                            # - repo의 데이터 생성 스크립트/README와 일치하도록 설정해야 함.
                                            # - 주로 root_path + video_path 형태로 실제 파일 경로를 구성.
	parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')     # '--sample_duration': LSTM/시퀀스에 넣을 프레임 수(시간적 길이).
                                                # - 보통 16, 8, 32 등으로 설정. 너무 작으면 시간 정보 부족, 너무 크면 메모리 증가.
                                                # - 정수여야 하며 type=int가 이를 보장(문자열 입력 시 에러).
	parser.add_argument(
        '--n_val_samples',
        default=3,
        type=int,
        help='Number of validation samples for each activity')  # '--n_val_samples': 클래스(액티비티)당 검증 샘플 수.
                                                                # - 예: 각 클래스에서 3 샘플을 고정으로 추출해 validation 수행.
                                                                # - 값이 0 또는 음수이면(논리적으로) 오류가 발생할 수 있으니 검증 필요.
	parser.add_argument(
		'--annotation_path',
		default='kinetics.json',
		type=str,
		help='Annotation file path')    # '--annotation_path': 라벨/메타데이터가 들어있는 annotation 파일 경로(예: JSON).
                                        # - 파일 포맷(예: Kinetics/ActivityNet 형식)을 코드가 기대하므로 포맷 확인 필요.
                                        # - 기본값이 kinetics.json인데 데이터셋에 따라 적절히 바꿀 것.
	parser.add_argument(
		'--gpu',
		default=0,
		type=int)  # '--gpu': 사용할 GPU 인덱스(정수). 예: 0, 1, 2 ...
                   # - 주의: 이것은 단순 정수 인자이므로 `--gpu` 만 쓰면 에러(값 필요). 플래그로 쓰려면 action을 달아야 함.
                   # - 멀티GPU 지정(예: "0,1")을 지원하려면 문자열로 받아 파싱하거나 nargs/型 변환을 추가해야 함.
                   # - 값 검증(음수/존재하지 않는 GPU 인덱스)은 이후 코드에서 처리 필요.
	parser.add_argument(
		'--sample_size',
		default=150,
		type=int,
		help='Height and width of inputs')  # '--sample_size': 프레임을 리사이즈할 때의 정사각형 한 변 길이(픽셀).
                                            # - 종종 모델 입력 크기와 일치시켜야 함(예: ResNet 입력 크기).
                                            # - 홀수/짝수 제한은 보통 없지만, 모델 아키텍처에 따라 요구될 수 있음.
	parser.add_argument(
		'--log_interval',
		default=10,
		type=int,
		help='Log interval for showing training loss')  # '--log_interval': 몇 배치마다 로그를 찍을지(정수).
                                                         # - "배치 단위"로 해석되는 경우가 많음(코드의 사용처 확인 필요).
                                                         # - 너무 자주 로그를 찍으면 I/O 비용 발생, 너무 드물면 학습 상황 파악이 어려움.
	parser.add_argument(
		'--save_interval',
		default=2,
		type=int,
		help='Model saving interval')  # '--save_interval': 모델을 저장하는 주기(에폭 단위일 가능성 높음).
                                       # - 사용부 코드(train.py 등)에서 "if epoch % save_interval == 0"과 같이 사용되는지 확인.
                                       # - 대규모 학습 시 디스크 공간 고려 필요.
	parser.add_argument(
        '--model',
        default='cnnlstm',
        type=str,
        help=
        '(cnnlstm | cnnlstm_attn |')  # '--model': 사용할 모델 이름(문자열). 기본 'cnnlstm'.
                                      # - 도움말 문자열이 닫히지 않음: 실제로는 '(cnnlstm | cnnlstm_attn | ...)' 같은 형식이어야 함.
                                      # - 권장 개선: add_argument(..., choices=['cnnlstm','cnnlstm_attn', ...]) 로 허용값을 제한하면 사용자 실수 방지.
                                      # - 모델 이름은 model.py에서 해당 토폴로지를 찾을 수 있도록 코드와 일치시켜야 함.
	parser.add_argument(
        '--n_classes',
        default=400,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)'
    )  # '--n_classes': 분류할 클래스 수(정수). dataset에 맞춰 정확히 지정해야 downstream loss/출력 크기와 일치함.
       # - 기본값 400은 Kinetics에 맞춘 값으로 보임. 잘못 설정시 모델 출력 크기 불일치로 실패 가능.
	parser.add_argument(
		'--lr_rate',
		default=1e-3,
		type=float,
		help='Initial learning rate (divided by 10 while training by lr scheduler)')  # '--lr_rate': 초깃값 학습률 (float).
                                                                                      # - 학습률은 매우 민감한 하이퍼파라미터.
                                                                                      # - argparse의 type=float은 문자열 -> float 변환; 부동소수점 표기 방식(예: 1e-3)도 허용.
	parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')  # '--momentum': SGD에서 사용하는 모멘텀 값(0~1 범위가 일반적).
                                                                                 # - 너무 높으면 학습이 발산할 수 있고, 너무 낮으면 수렴이 느릴 수 있음.
                                                                                 # - momentum은 optimizer 설정 시 그대로 전달됨.
	parser.add_argument(
		'--dampening', default=0.9, type=float, help='dampening of SGD')  # '--dampening': SGD의 dampening 파라미터.
                                                                           # - dampening은 momentum 효과를 감쇠시키는 용도(보통 0.0이 기본).
                                                                           # - 여기서 0.9라는 값은 흔한 관례(대부분 0.0)와 다르므로 실수 가능성 있음. 확인 권장.
	parser.add_argument(
		'--weight_decay', default=1e-3, type=float, help='Weight Decay')  # '--weight_decay': 가중치 감쇠(L2 정규화) 계수.
                                                                           # - 일반적으로 1e-4 ~ 1e-2 범위에서 실험. 너무 크면 과소적합 발생.
                                                                           # - 옵티마이저 생성 시 optimizer = SGD(..., weight_decay=args.weight_decay)
	parser.add_argument(
		'--no_mean_norm',
		action='store_true',
		help='If true, inputs are not normalized by mean.')  # '--no_mean_norm': 플래그형 옵션.
                                                           # - action='store_true'이면 CLI에서 이 플래그가 주어질 때 args.no_mean_norm == True가 됨.
                                                           # - 플래그를 주지 않으면 False. (기본값을 명시하려면 set_defaults 사용)
                                                           # - 의미: 보통 mean subtraction(채널별 평균 빼기)을 하지 않겠다는 옵션.
	parser.set_defaults(no_mean_norm=False)  # no_mean_norm의 기본값을 False로 명시적으로 설정.
                                             # - set_defaults는 parser에 기본 Namespace 값을 넣는 역할(나중에 add_argument로 같은 dest가 추가돼도 적용됨).
                                             # - 순서(앞/뒤)에 상관없이 동작하지만, 가독성을 위해 관련 add_argument 뒤에 두는 게 명확함.
	parser.add_argument(
        '--mean_dataset',
        default='activitynet',
        type=str,
        help=
        'dataset for mean values of mean subtraction (activitynet | kinetics)')  # '--mean_dataset': mean subtraction에 사용할 사전 계산된 평균값 세트 선택.
                                                                               # - 예: activitynet/kinetics에 대한 channel mean 값이 미리 계산되어 있고,
                                                                               #       데이터 로딩 시 해당 평균을 빼는 데 사용됨.
                                                                               # - 입력값은 파일명 또는 키로 해석될 수 있으므로, 사용되는 코드와 형식 일치 필요.
	parser.add_argument(
		'--use_cuda',
		action='store_true',
		help='If true, use GPU.')  # '--use_cuda' 플래그: 이 플래그가 있으면 CUDA 사용(True), 없으면 False.
                                  # - 주의: --gpu 인자와 중복되는 개념일 수 있음(둘을 함께 쓰는 방식을 명확히 하자).
                                  # - 보통은 gpu 인덱스가 주어지면 use_cuda를 자동 True로 하는 편이 더 직관적.
	parser.set_defaults(std_norm=False)  # std_norm의 기본값을 False로 설정.
                                         # - 여기서는 std_norm add_argument가 아직 정의되기 전에 기본값을 넣어두고 있음.
                                         # - 동작상 문제는 없으나, 코드 가독성 차원에서 관련 add_argument 근처에 두는 것이 좋다.
	parser.add_argument(
		'--nesterov', action='store_true', help='Nesterov momentum')  # '--nesterov' 플래그: Nesterov 모멘텀 사용 여부.
                                                                          # - Nesterov를 사용하려면 일반적으로 momentum도 함께 설정해야 의미가 있음.
                                                                          # - optimizer 생성 시 nesterov=args.nesterov 로 전달.
	parser.set_defaults(nesterov=False)  # nesterov 기본값을 False로 설정.
                                         # - store_true 플래그가 없을 때(사용자가 지정하지 않을 때) 기본값은 False가 되도록 명확히 함.
	parser.add_argument(
		'--optimizer',
		default='sgd',
		type=str,
		help='Currently only support SGD')  # '--optimizer': 최적화 알고리즘 이름(보통 'sgd' 또는 'adam' 등).
                                            # - 현재 코드가 SGD만 지원한다면 help에 그렇게 적어두는 건 맞지만,
                                            #   choices=['sgd'] 처럼 제한을 둘 수도 있음(추후 확장 용이).
	parser.add_argument(
		'--lr_patience',
		default=10,
		type=int,
		help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
	)   # '--lr_patience': ReduceLROnPlateau 같은 스케줄러에서 patience 값(반응을 기다리는 epoch 수).
        # - 보통 '에폭' 단위로 측정됨(사용 코드 확인 필요). 너무 작으면 학습 초기에 lr이 너무 빨리 줄어듦.
	parser.add_argument(
		'--batch_size', default=32, type=int, help='Batch Size')    # '--batch_size': 학습/검증 시 미니배치 크기.
                                                                    # - GPU 메모리에 따라 조정. 큰 값은 학습 안정성/속도에 영향.
                                                                    # - DataLoader에 전달되는 값과 일치해야 함.
	parser.add_argument(
		'--n_epochs',
		default=10,
		type=int,
		help='Number of total epochs to run')   # '--n_epochs': 전체 학습 반복 횟수.
                                                # - start_epoch, resume_path와 함께 사용되어 체크포인트 복원/계속 학습 제어.
	parser.add_argument(
		'--start_epoch',
		default=1,
		type=int,
		help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
	)   # '--start_epoch': 학습을 시작할 에폭 인덱스(주로 resume 시 사용).
        # - resume할 때 checkpoint의 epoch+1을 start_epoch로 주는 패턴이 흔함.
	parser.add_argument(
		'--resume_path',
		type=str,
		help='Resume training')  # '--resume_path': 체크포인트 파일(.pth 등)의 경로(문자열).
                                # - 지정하지 않으면 None(혹은 빈 문자열)이고 새로 학습 시작.
                                # - resume 동작은 저장된 optimizer state, epoch 등을 복원하도록 구현해야 함.
	parser.add_argument(
		'--pretrain_path', default='', type=str, help='Pretrained model (.pth)')    # '--pretrain_path': 사전 학습된 모델 파일 경로.
                                                                                    # - resume_path와 다르게, pretrain은 가중치 초기화 용도로 사용(optimizer state 복원 안함).
                                                                                    # - 빈 문자열 ''을 기본값으로 두었으므로 코드에서 빈 문자열 체크 필요.
	parser.add_argument(
		'--num_workers',
		default=4,
		type=int,
		help='Number of threads for multi-thread loading')  # '--num_workers': DataLoader의 워커 수.
                                                            # - 0일 경우 메인 프로세스에서 데이터 로딩(디버깅용).
                                                            # - Windows 환경에서는 muliprocessing spawn 문제 때문에 0~1 권장.
                                                            # - 너무 높으면 CPU 컨텍스트 스위칭/메모리 사용 증가.
	parser.add_argument(
		'--norm_value',
		default=1,
		type=int,
		help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')  # '--norm_value': 입력 정규화 스케일 값.
                                                                                    # - 관례: norm_value를 나눗셈 값으로 사용함.
                                                                                    #   예: img = img / args.norm_value
                                                                                    # - norm_value==1 => 값 범위 [0,255]; norm_value==255 => [0,1].
                                                                                    # - 주의: 이름이 직관적이지 않음(보통 divisor 혹은 scale_factor라 명명).
	parser.add_argument(
		'--std_norm',
		action='store_true',
		help='If true, inputs are normalized by standard deviation.')   # '--std_norm': 표준편차로 정규화 여부 플래그.
                                                                        # - 보통 mean subtraction 후 channel-wise std로 나눔.
                                                                        # - std_norm과 no_mean_norm의 조합에 따라 전처리 파이프라인이 달라지므로 문서화 필요.
																		
    # args는 단순한 딕셔너리도 아니고, 리스트도 아니고, argparse.Namespace 객체 
    #   =  Namespace(batch_size=32, lr_rate=0.001)   이렇게 생김     
    #   print(args.batch_size)  # 32     (이런 식으로 꺼낼 수 있음)
    #   print(args.lr_rate)     # 0.001  (이런 식으로 꺼낼 수 있음)                  
	args = parser.parse_args()  # 실제로 커맨드라인 인자들을 파싱.
                                # - 기본적으로 sys.argv[1:]를 해석.
                                # - 타입 변환 실패나 미지정 옵션(unknown) 사용 시 argparse가 에러 메시지와 함께 프로그램을 종료시킴.
                                # - 필요하면 parse_known_args()로 알 수 없는 인자를 무시하거나, 예외 처리를 추가할 수 있음.

	return args  # 파싱된 Namespace 반환. 호출부에서 args.whatever 로 접근하여 설정 사용.
