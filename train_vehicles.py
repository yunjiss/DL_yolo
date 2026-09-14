"""
실제 vehicles.v2-release.yolov8 (Roboflow) 데이터셋으로 YOLOv2 스타일 모델을 학습.

train.py(합성 도형 데모)와 구조는 동일하되, 데이터셋/설정만 vehicles_dataset.py +
vehicles_config.py로 교체했다.

실행 (Roboflow에서 다운받은 데이터셋 압축을 푼 폴더를 --data-root로 지정):
    python train_vehicles.py --data-root /path/to/vehicles.v2-release.yolov8 --epochs 50
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from model import YoloV2Tiny
from loss import YoloLoss
from vehicles_dataset import VehiclesDataset
from vehicles_config import NUM_ANCHORS, NUM_CLASSES


def train(
    data_root,
    epochs=50,
    batch_size=16,
    lr=1e-3,
    ckpt_path="checkpoints/yolo_vehicles.pt",
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = VehiclesDataset(os.path.join(data_root, "train"))
    val_ds = VehiclesDataset(os.path.join(data_root, "valid"))
    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = YoloV2Tiny(num_anchors=NUM_ANCHORS, num_classes=NUM_CLASSES).to(device)
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
    parser.add_argument("--data-root", required=True,
                         help="Roboflow 압축을 푼 폴더 (train/, valid/, test/ 가 있는 최상위 경로)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt", type=str, default="checkpoints/yolo_vehicles.pt")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ckpt_path=args.ckpt,
    )
