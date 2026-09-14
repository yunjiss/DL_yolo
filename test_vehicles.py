"""
실제 vehicles.v2-release.yolov8 데이터셋 샘플(vehicles_sample/)로
encode_targets -> (완벽한 예측 가정) -> decode_predictions -> NMS 라운드트립을 검증.
torch 없이 numpy만으로 실행 가능.
"""
import numpy as np

from utils import encode_targets, decode_predictions, non_max_suppression, iou_xyxy, xywh_to_xyxy
from vehicles_config import IMG_SIZE, S, ANCHORS, NUM_CLASSES, CLASS_NAMES
from vehicles_dataset import list_pairs, load_sample


def make_fake_prediction_from_target(target):
    pred = target.copy()
    class_onehot = pred[..., 5:]
    pred[..., 5:] = np.where(class_onehot > 0.5, 10.0, -10.0)
    return pred


def main():
    pairs = list_pairs("vehicles_sample")
    print(f"testing on {len(pairs)} real samples")

    total_gt, total_matched, total_collided = 0, 0, 0
    for image_path, label_path in pairs:
        image, boxes, labels = load_sample(image_path, label_path)
        if len(boxes) == 0:
            continue
        target = encode_targets(boxes, labels, S=S, anchors=ANCHORS, num_classes=NUM_CLASSES)
        n_encoded = int(target[..., 4].sum())
        total_collided += len(boxes) - n_encoded

        fake_pred = make_fake_prediction_from_target(target)
        detections = decode_predictions(
            fake_pred, anchors=ANCHORS, S=S, conf_thresh=0.5,
            img_size=IMG_SIZE, class_names=CLASS_NAMES,
        )
        detections = non_max_suppression(detections, iou_thresh=0.45)

        total_gt += len(boxes)
        gt_xyxy = np.array([xywh_to_xyxy(b) for b in boxes]) * IMG_SIZE

        for det in detections:
            ious = iou_xyxy(np.array([det["box"]]), gt_xyxy)[0]
            best = np.argmax(ious)
            if ious[best] > 0.9 and det["class_id"] == labels[best]:
                total_matched += 1

    print(f"\nTotal GT boxes           : {total_gt}")
    print(f"Grid/anchor 충돌로 누락    : {total_collided}  (같은 셀+앵커에 물체 2개 이상)")
    print(f"복원 가능한 GT 중 매칭됨   : {total_matched} / {total_gt - total_collided}")
    assert total_matched == total_gt - total_collided
    print("\n[PASS] 실제 vehicles 데이터로 encode -> decode -> NMS 라운드트립 검증 통과")
    print("(충돌분을 제외한 모든 GT 박스가 정확히 복원됨 — 인코딩/디코딩 로직이 실제 데이터에서도 올바름)")


if __name__ == "__main__":
    main()
