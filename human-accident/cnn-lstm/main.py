import os                           # 파일/폴더 경로 처리, 체크포인트 저장을 위해 사용
import time                         # 학습 시간 측정(로그 출력용)
import argparse                     # 커맨드라인 인자 파싱
from pathlib import Path            # 경로 해석 및 절대경로 변환

import torch                            # PyTorch 기본 패키지
import torch.nn as nn                   # 신경망 레이어/손실함수 등
from torch.optim import Adam            # Adam 옵티마이저
from torch.utils.data import DataLoader # 미니배치 데이터 로더
from torchvision import transforms      # 이미지 전처리(transform)

from train_valid_dataset import CustomData  # 우리가 만든 커스텀 데이터셋(Train/Valid 분할 포함)
from models.cnnlstm import CNNLSTM          # CNN + LSTM 모델 정의


def get_dataloaders(root: str, batch_size: int, num_workers: int, train_ratio: float, seed: int):
    """
    CustomData 기반으로 학습/검증 DataLoader를 생성합니다.
    - root        : 데이터 루트(클래스 폴더들이 바로 하위에 존재)
    - batch_size  : 배치 크기
    - num_workers : 데이터 로딩 서브프로세스 개수
    - train_ratio : train/valid 분할 비율(0.7이면 70:30)
    - seed        : 분할 셔플 시드(재현성)
    """

    # 프레임 단위 전처리 파이프라인: NumPy(RGB) -> PIL -> Resize -> Tensor -> Normalize(ImageNet)
    transform = transforms.Compose([
        transforms.ToPILImage(),                 # NumPy 이미지를 PIL 이미지로 변환
        transforms.Resize((224, 224)),           # 입력 해상도 통일
        transforms.ToTensor(),                   # (H,W,C)[0..255] -> (C,H,W)[0..1]
        transforms.Normalize(                    # ImageNet 사전학습 규약의 평균/표준편차로 정규화
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # 커스텀 데이터셋 생성: 내부에서 클래스별 스캔 및 70:30 분할 수행
    train_ds = CustomData(root=root, transform=transform, split="train", train_ratio=train_ratio, seed=seed)
    valid_ds = CustomData(root=root, transform=transform, split="valid", train_ratio=train_ratio, seed=seed)

    # DataLoader 생성: 학습은 셔플, 검증은 셔플하지 않음
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    
    return train_loader, valid_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    한 에폭 동안 학습을 수행합니다.
    - model     : 학습할 모델(CNNLSTM)
    - loader    : 학습 DataLoader
    - criterion : 손실함수(CrossEntropyLoss)
    - optimizer : 최적화기(Adam)
    - device    : 'cuda' 또는 'cpu'
    반환: (에폭 평균 손실, 에폭 평균 정확도)
    """
    model.train()                # 학습 모드
    running_loss = 0.0           # 손실 합계(평균 계산용)
    running_acc = 0.0            # 정답 개수 합계(정확도 계산용)
    total = 0                    # 샘플 수 합계

    for x, y in loader:          # 미니배치 반복: x=(B,T,C,H,W), y=(B,)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()     # 기울기 초기화
        logits = model(x)         # 순전파 -> (B, num_classes)
        loss = criterion(logits, y)  # 손실 계산
        loss.backward()           # 역전파
        optimizer.step()          # 가중치 업데이트

        bs = x.size(0)                            # 현재 배치 크기
        running_loss += loss.item() * bs          # 가중 합(평균용)
        preds = logits.argmax(dim=1)              # 예측 클래스 인덱스
        running_acc += (preds == y).sum().item()  # 정답 개수 누적
        total += bs

    # 전체 샘플 대비 평균 손실/정확도
    return (running_loss / total) if total else 0.0, (running_acc / total) if total else 0.0


def validate(model, loader, criterion, device):
    """
    검증(Validation) 1 에폭 수행(기울기 업데이트 없이 평가만).
    반환: (에폭 평균 손실, 에폭 평균 정확도)
    """
    model.eval()                 # 평가 모드
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    with torch.no_grad():        # 추론만 수행(메모리/연산 절약)
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.size(0)
            running_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            running_acc += (preds == y).sum().item()
            total += bs
    return (running_loss / total) if total else 0.0, (running_acc / total) if total else 0.0


def main():
    # ---------------------------------------------------------------------------
    """커맨드라인 인자를 받아 전체 학습 파이프라인을 실행합니다. main()에 매개변수를 주는 느낌입니다."""
    parser = argparse.ArgumentParser(description="CustomData + CNNLSTM simple training entry")
    # 데이터 경로/하이퍼파라미터 인자 정의
    parser.add_argument("--root", type=str, default=os.path.join("..", "data", "safety-data", "human-accident"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="snapshots")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=6)
    # (옵션) 간단 학습률 스케줄러 플래그
    parser.add_argument("--use-scheduler", action="store_true")
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)        # 체크포인트 저장 폴더 생성

    # ---------------------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")                        # 사용 디바이스 출력

    # ---------------------------------------------------------------------------
    # 🔥🔥🔥🔥🔥 여기서 root path를 지정해주고 있습니다.
    # --root 경로를 절대 경로로 해석하여 혼동 방지
    root_path = Path(args.root)
    if not root_path.is_absolute():
        root_path = (Path.cwd() / root_path).resolve()
    print(f"Resolved dataset root: {root_path}")

    if not root_path.exists():
        print(f"[오류] --root 경로가 존재하지 않습니다: {root_path}")
        print("경로를 확인하거나, 올바른 절대/상대 경로로 다시 실행하세요.")
        return

    # ---------------------------------------------------------------------------
    # 🔥🔥🔥🔥🔥 학습/검증 데이터로더 준비
    train_loader, valid_loader = get_dataloaders(
        root=str(root_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(f"Train samples: {len(train_loader.dataset)}, Valid samples: {len(valid_loader.dataset)}")

    # ---------------------------------------------------------------------------
    # 모델/손실/옵티마이저 정의
    model = CNNLSTM(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---------------------------------------------------------------------------
    # (옵션) ReduceLROnPlateau 스케줄러: 검증 손실(val_loss)이 감소하지 않으면 LR 감소
    scheduler = None
    if args.use_scheduler:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, verbose=True
        )

    # ---------------------------------------------------------------------------
    # 🔥🔥🔥🔥🔥 학습 루프
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # =========================================================================================================
        # 첫 배치에서 GPU 사용 여부를 확인하기 위한 디버그 출력
        printed_gpu_info = False

        def _train_one_epoch_with_debug(model, loader, criterion, optimizer, device):
            nonlocal printed_gpu_info
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            total = 0
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if not printed_gpu_info:
                    try:
                        model_device = next(model.parameters()).device
                        resnet_dev = model.resnet.fc[0].weight.device if hasattr(model.resnet, 'fc') else model_device
                        lstm_dev = model.lstm.weight_ih_l0.device
                        print("[디버그] 입력 x.device:", x.device)
                        print("[디버그] 모델 파라미터 device:", model_device)
                        print("[디버그] ResNet fc device:", resnet_dev)
                        print("[디버그] LSTM device:", lstm_dev)
                        if torch.cuda.is_available():
                            print("[디버그] CUDA name:", torch.cuda.get_device_name(0))
                            print("[디버그] CUDA memory allocated (MB):", round(torch.cuda.memory_allocated() / (1024**2), 2))
                    except Exception as e:
                        print(f"[디버그] 디바이스 확인 중 오류: {e}")
                    printed_gpu_info = True

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                bs = x.size(0)
                running_loss += loss.item() * bs
                preds = logits.argmax(dim=1)
                running_acc += (preds == y).sum().item()
                total += bs

            return (running_loss / total) if total else 0.0, (running_acc / total) if total else 0.0

        train_loss, train_acc = _train_one_epoch_with_debug(model, train_loader, criterion, optimizer, device)
        # =========================================================================================================

        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        took = time.time() - t0

        # 진행 상황 로그
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
            f"time={took:.1f}s"
        )

        if scheduler is not None:
            scheduler.step(val_loss)  # 스케줄러는 보통 검증 지표를 기준으로 동작

        # 매 에폭 체크포인트 저장
        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": args.num_classes,
        }
        ckpt_path = os.path.join(args.save_dir, f"CNNLSTM-epoch{epoch}-valacc{val_acc:.4f}.pth")
        torch.save(ckpt, ckpt_path)

        # 최고 성능(best) 모델 갱신
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))


if __name__ == "__main__":
    main()  # 스크립트 직접 실행 시 학습 시작
