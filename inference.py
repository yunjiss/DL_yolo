"""
추론 스크립트.

학습된 체크포인트를 불러와 이미지에 대해 forward pass -> 디코딩 -> NMS를 수행하고
결과를 박스로 그려서 저장한다.

실행 (합성 데이터로 테스트):
    python inference.py --ckpt checkpoints/yolo_tiny.pt --seed 999

내 이미지 파일로 테스트하려면:
    python inference.py --ckpt checkpoints/yolo_tiny.pt --image path/to/image.png
    (이미지는 128x128 RGB로 리사이즈되어 입력됩니다. 모델은 합성 도형 데이터로만
    학습되었으므로 실제 사진에는 의미 있는 탐지를 하지 못할 수 있습니다.)
"""
import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import YoloV2Tiny
from utils import decode_predictions, non_max_suppression, IMG_SIZE
from dataset import generate_sample, CLASS_COLORS


def load_model(ckpt_path, device="cpu"):
    model = YoloV2Tiny().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_image(model, image_np, device="cpu", conf_thresh=0.4, iou_thresh=0.45):
    """image_np: (H,W,3) uint8 -> NMS 적용된 detection list"""
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    image_t = image_t.to(device)
    with torch.no_grad():
        pred = model(image_t)[0].cpu().numpy()  # (S,S,A,5+C)
    detections = decode_predictions(pred, conf_thresh=conf_thresh)
    detections = non_max_suppression(detections, iou_thresh=iou_thresh)
    return detections


def draw_detections(image_np, detections, out_path, gt_boxes=None):
    img = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(img)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(det["class_id"], (255, 255, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{det['class_name']} {det['score']:.2f}"
        draw.text((x1, max(0, y1 - 10)), label, fill=color)

    img.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/yolo_tiny.pt")
    parser.add_argument("--image", default=None, help="입력 이미지 경로 (없으면 합성 샘플 사용)")
    parser.add_argument("--seed", type=int, default=999, help="합성 샘플 시드 (--image 미지정시)")
    parser.add_argument("--out", default="outputs/pred.png")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)

    if args.image:
        img = Image.open(args.image).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        image_np = np.array(img, dtype=np.uint8)
    else:
        image_np, gt_boxes, gt_labels = generate_sample(seed=args.seed)
        print(f"(합성 샘플, seed={args.seed}) ground truth objects: {len(gt_boxes)}")

    detections = predict_image(model, image_np, device, conf_thresh=args.conf, iou_thresh=args.iou)

    print(f"Detected objects: {len(detections)}")
    for det in detections:
        box = tuple(round(v, 1) for v in det["box"])
        print(f"  {det['class_name']:8s} score={det['score']:.3f} box={box}")

    out_path = draw_detections(image_np, detections, args.out)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
