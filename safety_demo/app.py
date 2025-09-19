import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import sys
from pathlib import Path

# 로컬 모듈 import
from cnnlstm_model import CNNLSTM
from torchvision import transforms
import config  # 설정 파일 import

# 페이지 설정
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide"
)

@st.cache_resource
def load_model(model_path):
    """모델을 로드하고 캐시합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() and not config.FORCE_CPU else "cpu")
    model = CNNLSTM(num_classes=config.NUM_CLASSES)
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # 체크포인트 구조 확인 및 처리
            if isinstance(checkpoint, dict):
                # 딕셔너리 형태인 경우, 'model_state_dict' 키가 있는지 확인
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # 키가 없으면 전체를 state_dict로 간주
                    model.load_state_dict(checkpoint)
            else:
                # 직접 state_dict인 경우
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            st.success(f"✅ 모델 로드 완료! (Device: {device})")
            return model, device
            
        except Exception as e:
            st.error(f"❌ 모델 로드 중 오류 발생: {str(e)}")
            
            # 체크포인트 구조 디버그 정보 표시
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict):
                    st.info(f"🔍 체크포인트 키들: {list(checkpoint.keys())}")
                    if 'model_state_dict' in checkpoint:
                        st.info("📦 'model_state_dict' 키 발견")
                    elif 'state_dict' in checkpoint:
                        st.info("📦 'state_dict' 키 발견")
                else:
                    st.info("📦 체크포인트는 직접 state_dict 형태입니다")
            except:
                pass
            
            return None, device
    else:
        st.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return None, device

def preprocess_video(video_path, fixed_len=None, target_fps=None):
    """비디오를 전처리하여 모델 입력 형태로 변환합니다."""
    
    # 기본값 설정
    if fixed_len is None:
        fixed_len = config.FIXED_SEQUENCE_LENGTH
    if target_fps is None:
        target_fps = config.TARGET_FPS
    
    # Transform 정의 (학습 시와 동일)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD
        ),
    ])
    
    # 비디오 읽기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("비디오를 열 수 없습니다.")
        return None
    
    # FPS 정보
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    st.info(f"📹 원본 FPS: {original_fps}, 총 프레임: {frame_count}")
    
    # 프레임 추출 (target_fps로 리샘플링)
    frames = []
    frame_interval = max(1, original_fps // target_fps)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            # BGR -> RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
        frame_idx += 1
    
    cap.release()
    
    if len(frames) == 0:
        st.error("프레임을 추출할 수 없습니다.")
        return None
    
    st.info(f"🎬 추출된 프레임 수: {len(frames)}")
    
    # 길이 고정 (16프레임)
    if len(frames) >= fixed_len:
        # 균등 샘플링
        indices = np.linspace(0, len(frames) - 1, fixed_len, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        # 패딩 (마지막 프레임 반복)
        while len(frames) < fixed_len:
            frames.append(frames[-1])
    
    # Transform 적용
    processed_frames = []
    for frame in frames:
        tensor_frame = transform(frame)  # (C, H, W)
        processed_frames.append(tensor_frame)
    
    # 텐서로 변환: (T, C, H, W)
    video_tensor = torch.stack(processed_frames)
    
    # 배치 차원 추가: (1, T, C, H, W)
    video_tensor = video_tensor.unsqueeze(0)
    
    return video_tensor, frames

def predict_video(model, video_tensor, device):
    """모델로 예측을 수행합니다."""
    with torch.no_grad():
        video_tensor = video_tensor.to(device)
        logits = model(video_tensor)  # (1, num_classes)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def main():
    st.title(config.PAGE_TITLE)
    st.markdown("---")
    
    # 사이드바 - 모델 설정
    st.sidebar.header("⚙️ 모델 설정")
    
    # 사용 가능한 모델 목록 표시
    available_models = config.get_available_models()
    
    if available_models:
        st.sidebar.subheader("📁 사용 가능한 모델")
        
        # 모델 선택 드롭다운
        model_options = [f"{m['name']} ({m['size_mb']}MB)" for m in available_models]
        selected_idx = st.sidebar.selectbox(
            "모델 선택",
            range(len(model_options)),
            format_func=lambda x: model_options[x],
            help="학습된 모델을 선택하세요"
        )
        model_path = available_models[selected_idx]['path']
        
        # 선택된 모델 정보 표시
        selected_model = available_models[selected_idx]
        st.sidebar.info(f"✅ 선택된 모델: {selected_model['name']}")
        
    else:
        st.sidebar.warning("⚠️ 사용 가능한 모델이 없습니다.")
        model_path = st.sidebar.text_input(
            "모델 파일 경로", 
            value=config.get_model_path(),
            help="학습된 .pth 파일의 경로를 입력하세요"
        )
    
    # 경로 상태 확인
    paths_status = config.validate_paths()
    with st.sidebar.expander("🔍 경로 상태 확인"):
        for path_name, exists in paths_status.items():
            status_icon = "✅" if exists else "❌"
            st.text(f"{status_icon} {path_name}: {'존재' if exists else '없음'}")
    
    # 모델 로드
    model, device = load_model(model_path)
    
    if model is None:
        st.warning("⚠️ 모델을 먼저 로드해주세요.")
        st.info("💡 학습이 완료된 후 `snapshots/best.pth` 파일이 생성됩니다.")
        
        # 학습 진행 안내
        with st.expander("🎓 모델 학습 방법"):
            st.markdown("""
            **1. 학습 실행 명령:**
            ```powershell
            cd human-accident\\cnn-lstm
            uv run python main.py --root "C:\\Users\\bb\\Desktop\\learn_deeplearning\\human-accident\\data\\safety-data\\human-accident" --epochs 10 --batch-size 8 --lr 1e-4 --weight-decay 1e-4 --num-workers 2 --train-ratio 0.7 --seed 42 --num-classes 6
            ```
            
            **2. 학습 완료 확인:**
            - `snapshots/` 폴더에 `best.pth` 파일이 생성됨
            - 콘솔에 "Best model saved!" 메시지 출력
            
            **3. 예상 학습 시간:**
            - GPU 사용 시: 약 30분-1시간 (에폭 수에 따라)
            - CPU 사용 시: 약 2-4시간
            
            **4. 학습 중 확인사항:**
            - tqdm 진행바로 실시간 진행상황 확인
            - GPU 사용 여부 디버그 로그 확인
            - 각 에폭별 train/val 성능 모니터링
            """)
        
        # 현재 snapshots 폴더 상태 표시
        snapshots_path = Path(config.get_snapshots_dir())
        if snapshots_path.exists():
            files = list(snapshots_path.glob("*"))
            st.info(f"📁 현재 snapshots 폴더: {len(files)}개 파일")
            if files:
                for file in files:
                    st.text(f"  - {file.name}")
        
        return
    
    # 메인 영역
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 비디오 업로드")
        
        uploaded_file = st.file_uploader(
            "분석할 비디오를 업로드하세요",
            type=config.SUPPORTED_VIDEO_FORMATS,
            help=f"지원 형식: {', '.join(config.SUPPORTED_VIDEO_FORMATS).upper()}"
        )
        
        if uploaded_file is not None:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name
            
            # 비디오 표시
            st.video(uploaded_file)
            
            # 분석 버튼
            if st.button("🔍 안전사고 분석 시작", type="primary"):
                with st.spinner("비디오 분석 중..."):
                    # 전처리
                    result = preprocess_video(temp_video_path)
                    if result is None:
                        return
                    
                    video_tensor, frames = result
                    
                    # 예측
                    pred_class, confidence, all_probs = predict_video(model, video_tensor, device)
                    
                    # 결과 표시
                    with col2:
                        st.header("📊 분석 결과")
                        
                        # 예측 결과
                        predicted_label_kr = config.CLASS_NAMES_KR[pred_class]
                        predicted_label_en = config.CLASS_NAMES[pred_class]
                        
                        if pred_class == 5:  # no-accident
                            st.success(f"✅ **{predicted_label_kr}**")
                            st.success(f"신뢰도: **{confidence:.2%}**")
                        else:
                            st.error(f"⚠️ **{predicted_label_kr} 감지!**")
                            st.error(f"신뢰도: **{confidence:.2%}**")
                        
                        st.info(f"영문 클래스: `{predicted_label_en}`")
                        
                        # 모든 클래스별 확률
                        st.subheader("📈 클래스별 확률")
                        
                        # 바 차트
                        st.bar_chart(dict(zip(config.CLASS_NAMES_KR, all_probs)))
                        
                        # 테이블
                        import pandas as pd
                        df = pd.DataFrame({
                            "클래스": config.CLASS_NAMES_KR,
                            "영문": config.CLASS_NAMES,
                            "확률": [f"{prob:.2%}" for prob in all_probs]
                        })
                        st.dataframe(df, width='stretch')
                        
                        # 샘플 프레임 표시
                        st.subheader("🎬 분석된 프레임 샘플")
                        
                        # 4개 프레임만 표시
                        sample_indices = [0, 5, 10, 15]
                        frame_cols = st.columns(4)
                        
                        for i, idx in enumerate(sample_indices):
                            if idx < len(frames):
                                with frame_cols[i]:
                                    st.image(
                                        frames[idx], 
                                        caption=f"프레임 {idx+1}",
                                        width='stretch'
                                    )
            
            # 임시 파일 정리
            try:
                os.unlink(temp_video_path)
            except:
                pass
    
    # 하단 정보
    st.markdown("---")
    with st.expander("ℹ️ 시스템 정보"):
        st.markdown(f"""
        **모델 정보:**
        - 아키텍처: CNN-LSTM
        - 백본: ResNet-101 (사전학습)
        - 입력 형태: 16프레임 × 224×224 RGB
        - 클래스 수: 6개
        - 디바이스: {device}
        
        **지원 사고 유형:**
        - 충돌 사고 (bump)
        - 넘어짐 사고 (fall-down)  
        - 추락 사고 (fall-off)
        - 타격 사고 (hit)
        - 끼임 사고 (jam)
        - 정상 상황 (no-accident)
        """)

if __name__ == "__main__":
    main()
