import os
import time
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from train_valid_dataset import CustomData
from models.cnnlstm import CNNLSTM


def get_dataloaders(root: str, batch_size: int, num_workers: int, train_ratio: float, seed: int):
    """CustomData 기반 train/valid DataLoader 생성"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = CustomData(root=root, transform=transform, split="train", train_ratio=train_ratio, seed=seed)
    valid_ds = CustomData(root=root, transform=transform, split="valid", train_ratio=train_ratio, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)
    
    return train_loader, valid_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

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


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    with torch.no_grad():
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
    parser = argparse.ArgumentParser(description="CustomData + CNNLSTM simple training entry")
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
    # Optional simple LR scheduler flag (off by default)
    parser.add_argument("--use-scheduler", action="store_true")
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, valid_loader = get_dataloaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(f"Train samples: {len(train_loader.dataset)}, Valid samples: {len(valid_loader.dataset)}")

    model = CNNLSTM(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Optional LR scheduler (ReduceLROnPlateau)
    scheduler = None
    if args.use_scheduler:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, verbose=True)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        took = time.time() - t0

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
            f"time={took:.1f}s"
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        # Save checkpoint every epoch
        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": args.num_classes,
        }
        ckpt_path = os.path.join(args.save_dir, f"CNNLSTM-epoch{epoch}-valacc{val_acc:.4f}.pth")
        torch.save(ckpt, ckpt_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))


if __name__ == "__main__":
    main()
