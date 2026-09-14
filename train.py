"""
학습 스크립트.

합성(더미) 데이터셋(dataset.YoloDataset)으로 YoloV2Tiny 모델을 학습한다.
실행:
    python train.py --epochs 30 --batch-size 16
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from model import YoloV2Tiny
from loss import YoloLoss
from dataset import YoloDataset


def train(
    epochs=30,
    batch_size=16,
    lr=1e-3,
    train_samples=512,
    val_samples=64,
    ckpt_path="checkpoints/yolo_tiny.pt",
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = YoloDataset(num_samples=train_samples, seed_offset=0)
    val_ds = YoloDataset(num_samples=val_samples, seed_offset=1_000_000)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = YoloV2Tiny().to(device)
    criterion = YoloLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"total": 0.0, "coord": 0.0, "obj": 0.0, "noobj": 0.0, "class": 0.0}
        n_batches = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            loss, loss_dict = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in running:
                running[k] += loss_dict[k]
            n_batches += 1

        train_msg = " ".join(f"{k}={v / n_batches:.4f}" for k, v in running.items())

        # --- validation ---
        model.eval()
        val_running = {"total": 0.0, "coord": 0.0, "obj": 0.0, "noobj": 0.0, "class": 0.0}
        n_val_batches = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                _, loss_dict = criterion(preds, targets)
                for k in val_running:
                    val_running[k] += loss_dict[k]
                n_val_batches += 1
        val_msg = " ".join(f"{k}={v / n_val_batches:.4f}" for k, v in val_running.items())

        print(f"[epoch {epoch:03d}/{epochs}] train: {train_msg} | val: {val_msg}")

        val_total = val_running["total"] / n_val_batches
        if val_total < best_val:
            best_val = val_total
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"  -> best checkpoint saved (val total={val_total:.4f}) at {ckpt_path}")

    print("training done. best val loss:", best_val)
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--ckpt", type=str, default="checkpoints/yolo_tiny.pt")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        ckpt_path=args.ckpt,
    )
