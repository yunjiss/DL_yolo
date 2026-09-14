"""
Roboflow YOLOv8 포맷(images/*.jpg + labels/*.txt, 한 줄당 `class cx cy w h` 정규화)
데이터셋 로더. utils.py의 encode_targets/기존 모델·손실함수를 그대로 재사용하고,
설정값만 vehicles_config.py(실제 차량 데이터셋에 맞춘 IMG_SIZE/S/ANCHORS/클래스)를 쓴다.

디렉토리 구조 (Roboflow가 주는 그대로):
    <root>/train/images/*.jpg, <root>/train/labels/*.txt
    <root>/valid/images/*.jpg, <root>/valid/labels/*.txt
    <root>/test/images/*.jpg,  <root>/test/labels/*.txt
"""
import os
import glob

import numpy as np
from PIL import Image

from utils import encode_targets
from vehicles_config import IMG_SIZE, S, ANCHORS, NUM_CLASSES, CLASS_NAMES


def list_pairs(split_root):
    """split_root 예: VOCdata_vehicles/train -> (image_path, label_path) 리스트"""
    img_dir = os.path.join(split_root, "images")
    lbl_dir = os.path.join(split_root, "labels")
    pairs = []
    for img_path in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))) + \
                    sorted(glob.glob(os.path.join(img_dir, "*.png"))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        if os.path.exists(lbl_path):
            pairs.append((img_path, lbl_path))
    return pairs


def parse_yolo_label(label_path):
    """`class cx cy w h` (정규화, 0~1) 한 줄당 객체 하나. 이미 앵커/그리드와 무관하게
    전체 이미지 기준 정규화 좌표라서 이미지 리사이즈와 상관없이 그대로 유효하다."""
    boxes, labels = [], []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((cx, cy, w, h))
            labels.append(cls)
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_sample(image_path, label_path, img_size=IMG_SIZE):
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    image = np.array(img, dtype=np.uint8)
    boxes, labels = parse_yolo_label(label_path)
    return image, boxes, labels


# --- torch 래퍼 (학습에 사용, torch가 설치된 환경에서만 임포트됨) ---
try:
    import torch
    from torch.utils.data import Dataset

    class VehiclesDataset(Dataset):
        def __init__(self, split_root, img_size=IMG_SIZE):
            self.pairs = list_pairs(split_root)
            if not self.pairs:
                raise FileNotFoundError(f"No image/label pairs found under {split_root}")
            self.img_size = img_size

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            image_path, label_path = self.pairs[idx]
            image, boxes, labels = load_sample(image_path, label_path, self.img_size)
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

            if len(boxes) == 0:
                target_np = np.zeros((S, S, len(ANCHORS), 5 + NUM_CLASSES), dtype=np.float32)
            else:
                target_np = encode_targets(
                    boxes, labels, S=S, anchors=ANCHORS, num_classes=NUM_CLASSES
                )
            target_t = torch.from_numpy(target_np)
            return image_t, target_t

except ImportError:
    VehiclesDataset = None


if __name__ == "__main__":
    # torch 없이 실행 가능한 단독 검증: 파싱 + 타겟 인코딩이 정상 동작하는지 확인
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "vehicles_sample"
    pairs = list_pairs(root)
    print(f"found {len(pairs)} pairs under {root}")
    assert len(pairs) > 0, "샘플 데이터가 없습니다"

    total_boxes = 0
    total_encoded = 0
    for image_path, label_path in pairs:
        image, boxes, labels = load_sample(image_path, label_path)
        assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
        if len(boxes) == 0:
            continue
        assert labels.max() < NUM_CLASSES and labels.min() >= 0, "라벨 인덱스 범위 오류"
        target = encode_targets(boxes, labels, S=S, anchors=ANCHORS, num_classes=NUM_CLASSES)
        total_boxes += len(boxes)
        total_encoded += int(target[..., 4].sum())

    print(f"total GT boxes: {total_boxes}")
    print(f"total encoded (obj=1) cells: {total_encoded}")
    print(f"충돌(같은 셀+앵커에 물체 2개 이상 배정)로 인한 손실: {total_boxes - total_encoded}개")
