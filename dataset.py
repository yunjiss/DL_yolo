"""
합성(더미) 데이터셋: 검은 배경 위에 원(circle) / 사각형(square) / 삼각형(triangle)을
무작위 위치·크기·색상으로 그려서 학습/추론 코드를 검증하기 위한 데이터셋.

- generate_sample(): torch 없이 numpy/PIL만으로 동작 (단독 테스트 가능)
- YoloDataset: torch.utils.data.Dataset 래퍼 (학습 시 사용)
"""
import random
import numpy as np
from PIL import Image, ImageDraw

from utils import IMG_SIZE, S, ANCHORS, NUM_CLASSES, CLASS_NAMES, encode_targets

# 클래스별 색상 (시각적 구분용, 탐지에는 색상을 쓰지 않음 - 모양만으로 구분)
CLASS_COLORS = {
    0: (220, 60, 60),    # circle - red
    1: (60, 140, 220),   # square - blue
    2: (60, 200, 100),   # triangle - green
}


def _random_box(img_size, min_size=16, max_size=48):
    w = random.randint(min_size, max_size)
    h = random.randint(min_size, max_size)
    x1 = random.randint(0, img_size - w)
    y1 = random.randint(0, img_size - h)
    return x1, y1, x1 + w, y1 + h


def _boxes_overlap(b1, b2, margin=2):
    x1, y1, x2, y2 = b1
    a1, b1_, a2, b2_ = b2
    return not (x2 + margin < a1 or a2 + margin < x1 or y2 + margin < b1_ or b2_ + margin < y1)


def generate_sample(img_size=IMG_SIZE, min_objs=1, max_objs=3, seed=None):
    """무작위 도형 이미지 1장 생성.
    반환: image (H,W,3) uint8 numpy array, boxes (N,4) normalized cxcywh, labels (N,)
    """
    if seed is not None:
        random.seed(seed)

    img = Image.new("RGB", (img_size, img_size), (15, 15, 15))
    draw = ImageDraw.Draw(img)

    n_objs = random.randint(min_objs, max_objs)
    placed = []
    boxes, labels = [], []

    attempts = 0
    while len(placed) < n_objs and attempts < 50:
        attempts += 1
        box = _random_box(img_size)
        if any(_boxes_overlap(box, p) for p in placed):
            continue
        placed.append(box)

        cls = random.randint(0, NUM_CLASSES - 1)
        color = CLASS_COLORS[cls]
        x1, y1, x2, y2 = box

        if cls == 0:  # circle
            draw.ellipse([x1, y1, x2, y2], fill=color)
        elif cls == 1:  # square/rectangle
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:  # triangle
            draw.polygon([(x1, y2), ((x1 + x2) / 2, y1), (x2, y2)], fill=color)

        cx = (x1 + x2) / 2 / img_size
        cy = (y1 + y2) / 2 / img_size
        w = (x2 - x1) / img_size
        h = (y2 - y1) / img_size
        boxes.append((cx, cy, w, h))
        labels.append(cls)

    image = np.array(img, dtype=np.uint8)
    return image, np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


# --- torch 래퍼 (학습에 사용, torch가 설치된 환경에서만 임포트됨) ---
try:
    import torch
    from torch.utils.data import Dataset

    class YoloDataset(Dataset):
        def __init__(self, num_samples=512, img_size=IMG_SIZE, seed_offset=0):
            self.num_samples = num_samples
            self.img_size = img_size
            self.seed_offset = seed_offset

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # 매 epoch 동일한 idx가 같은 이미지가 되도록 idx 기반 seed 사용
            image, boxes, labels = generate_sample(
                self.img_size, seed=idx + self.seed_offset
            )
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

            if len(boxes) == 0:
                target_np = np.zeros((S, S, len(ANCHORS), 5 + NUM_CLASSES), dtype=np.float32)
            else:
                target_np = encode_targets(boxes, labels)
            target_t = torch.from_numpy(target_np)

            return image_t, target_t

except ImportError:
    # torch가 없는 환경(예: 이 검증용 샌드박스)에서도 generate_sample()은 그대로 쓸 수 있음
    YoloDataset = None


if __name__ == "__main__":
    # 단독 실행: 합성 데이터 생성 + 인코딩 로직을 torch 없이 검증
    img, boxes, labels = generate_sample(seed=0)
    print("image shape:", img.shape, img.dtype)
    print("boxes:\n", boxes)
    print("labels:", labels, [CLASS_NAMES[l] for l in labels])

    target = encode_targets(boxes, labels)
    print("target shape:", target.shape)
    n_obj_cells = int(target[..., 4].sum())
    print("encoded object count (should equal len(boxes)):", n_obj_cells, "vs", len(boxes))
