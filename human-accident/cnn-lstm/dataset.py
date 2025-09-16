# from datasets.kinetics import Kinetics      # 주석 처리됨: Kinetics 데이터셋 클래스 임포트
# from datasets.activitynet import ActivityNet  # 주석 처리됨: ActivityNet 데이터셋 클래스 임포트
from datasets.ucf101 import UCF101          # datasets/ucf101.py 파일에서 UCF101 데이터셋 클래스를 임포트
# from datasets.hmdb51 import HMDB51        # 주석 처리됨: HMDB51 데이터셋 클래스 임포트


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):  # 학습용 데이터셋 객체를 생성하여 반환하는 함수
    # opt: 커맨드라인 옵션 객체
    # spatial_transform: 이미지 프레임에 적용될 공간적 변환 (e.g., 리사이즈, 정규화)
    # temporal_transform: 비디오 시퀀스에 적용될 시간적 변환 (e.g., 랜덤 크롭)
    # target_transform: 라벨 데이터에 적용될 변환
    
    if opt.dataset == 'ucf101':  # 옵션으로 받은 데이터셋 이름이 'ucf101'인 경우
        training_data = UCF101(  # UCF101 클래스의 인스턴스를 생성
            opt.video_path,                  # 비디오(프레임)가 저장된 경로
            opt.annotation_path,             # 어노테이션 파일(라벨 정보) 경로
            'training',                      # 데이터셋의 종류를 '학습용'으로 지정
            spatial_transform=spatial_transform,    # 전달받은 공간적 변환 적용
            temporal_transform=temporal_transform,  # 전달받은 시간적 변환 적용
            target_transform=target_transform)      # 전달받은 타겟 변환 적용

    return training_data  # 생성된 학습 데이터셋 객체를 반환

def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):  # 검증용 데이터셋 객체를 생성하여 반환하는 함수
    # 함수의 파라미터들은 get_training_set과 동일
    
    if opt.dataset == 'ucf101':  # 옵션으로 받은 데이터셋 이름이 'ucf101'인 경우
        validation_data = UCF101(  # UCF101 클래스의 인스턴스를 생성
            opt.video_path,                      # 비디오(프레임)가 저장된 경로
            opt.annotation_path,                 # 어노테이션 파일(라벨 정보) 경로
            'validation',                        # 데이터셋의 종류를 '검증용'으로 지정
            opt.n_val_samples,                   # 검증에 사용할 클래스당 샘플 수
            spatial_transform,                   # 전달받은 공간적 변환 적용
            temporal_transform,                  # 전달받은 시간적 변환 적용
            target_transform,                    # 전달받은 타겟 변환 적용
            sample_duration=opt.sample_duration) # 비디오 클립의 길이(프레임 수)
    return validation_data  # 생성된 검증 데이터셋 객체를 반환


# def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
#     assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']
#     assert opt.test_subset in ['val', 'test']

#     if opt.test_subset == 'val':
#         subset = 'validation'
#     elif opt.test_subset == 'test':
#         subset = 'testing'
#     if opt.dataset == 'kinetics':
#         test_data = Kinetics(
#             opt.video_path,
#             opt.annotation_path,
#             subset,
#             0,
#             spatial_transform,
#             temporal_transform,
#             target_transform,
#             sample_duration=opt.sample_duration)
#     elif opt.dataset == 'activitynet':
#         test_data = ActivityNet(
#             opt.video_path,
#             opt.annotation_path,
#             subset,
#             True,
#             0,
#             spatial_transform,
#             temporal_transform,
#             target_transform,
#             sample_duration=opt.sample_duration)
#     elif opt.dataset == 'ucf101':
#         test_data = UCF101(
#             opt.video_path,
#             opt.annotation_path,
#             subset,
#             0,
#             spatial_transform,
#             temporal_transform,
#             target_transform,
#             sample_duration=opt.sample_duration)
#     elif opt.dataset == 'hmdb51':
#         test_data = HMDB51(
#             opt.video_path,
#             opt.annotation_path,
#             subset,
#             0,
#             spatial_transform,
#             temporal_transform,
#             target_transform,
#             sample_duration=opt.sample_duration)

#     return test_data

# 주석 처리된 get_test_set 함수: 테스트 데이터셋을 생성하는 로직. 현재 프로젝트에서는 사용되지 않음.
