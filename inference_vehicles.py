"""
vehicles 데이터셋으로 학습한 체크포인트로 추론.

실행:
    python inference_vehicles.py --ckpt checkpoints/yolo_vehicles.pt \
        --data-root /path/to/vehicles.v2-release.yolov8 --split test --index 0

내 이미지로:
    python inference_vehicles.py --ckpt checkpoints/yolo_vehicles.pt --image my.jpg
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import YoloV2Tiny
from utils import decode_predictions, non_max_suppression
from vehicles_config import IMG_SIZE, S, ANCHORS, NUM_ANCHORS, NUM_CLASSES, CLASS_NAMES
from vehicles_dataset import list_pairs, load_sample


def load_model(ckpt_path, device="cpu"):
    model = YoloV2Tiny(num_anchors=NUM_ANCHORS, num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_image(model, image_np, device="cpu", conf_thresh=0.4, iou_thresh=0.45):
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    image_t = image_t.to(device)
    with torch.no_grad():
        pred = model(image_t)[0].cpu().numpy()
    detections = decode_predictions(
        pred, anchors=ANCHORS, S=S, conf_thresh=conf_thresh,
        img_size=IMG_SIZE, class_names=CLASS_NAMES,
    )
    detections = non_max_suppression(detections, iou_thresh=iou_thresh)
    return detections


def draw_detections(image_np, detections, out_path):
    img = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 120), width=2)
        draw.text((x1, max(0, y1 - 10)), f"{det['class_name']} {det['score']:.2f}", fill=(0, 255, 120))
    img.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/yolo_vehicles.pt")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image", default=None)
    parser.add_argument("--out", default="outputs/pred_vehicles.png")
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)

    if args.image:
        img = Image.open(args.image).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        image_np = np.array(img, dtype=np.uint8)
    else:
        assert args.data_root, "--image 가 없으면 --data-root 를 지정해야 합니다"
        pairs = list_pairs(os.path.join(args.data_root, args.split))
        image_np, gt_boxes, gt_labels = load_sample(*pairs[args.index])
        print(f"ground truth objects: {len(gt_boxes)}")

    detections = predict_image(model, image_np, device, conf_thresh=args.conf)

    print(f"Detected objects: {len(detections)}")
    for det in detections:
        box = tuple(round(v, 1) for v in det["box"])
        print(f"  {det['class_name']:12s} score={det['score']:.3f} box={box}")

    out_path = draw_detections(image_np, detections, args.out)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
