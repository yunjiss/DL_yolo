"""
torch 없이 numpy만으로 encode -> decode -> NMS 전체 파이프라인을 검증하는 스크립트.
GT 박스를 encode_targets로 인코딩한 뒤, 그 값을 '모델이 완벽히 예측했다'고 가정하고
decode_predictions로 복원해서 원본 박스와 거의 일치하는지 확인한다.
"""
import numpy as np
from utils import (
    IMG_SIZE, S, ANCHORS, NUM_CLASSES,
    encode_targets, decode_predictions, non_max_suppression,
    iou_xyxy, xywh_to_xyxy,
)
from dataset import generate_sample


def make_fake_prediction_from_target(target):
    """target(정답 인코딩)을 그대로 '모델 출력'처럼 변환.
    obj가 1인 위치는 그대로 두고, class one-hot -> 큰 logit으로 변환(softmax 후 거의 1이 되도록).
    obj가 0인 위치는 confidence를 낮게(0) 유지.
    """
    pred = target.copy()
    class_onehot = pred[..., 5:]
    # one-hot(0/1) -> logit(-10/10) 로 변환해서 softmax 후 거의 one-hot이 되게 함
    pred[..., 5:] = np.where(class_onehot > 0.5, 10.0, -10.0)
    return pred


def main():
    total_matched = 0
    total_gt = 0
    max_center_err = 0.0
    max_size_err = 0.0

    n_trials = 20
    for trial in range(n_trials):
        image, boxes, labels = generate_sample(seed=100 + trial)
        if len(boxes) == 0:
            continue
        target = encode_targets(boxes, labels)
        fake_pred = make_fake_prediction_from_target(target)

        detections = decode_predictions(fake_pred, conf_thresh=0.5)
        detections = non_max_suppression(detections, iou_thresh=0.45)

        total_gt += len(boxes)
        gt_xyxy = np.array([xywh_to_xyxy(b) for b in boxes]) * IMG_SIZE

        for det in detections:
            ious = iou_xyxy(np.array([det["box"]]), gt_xyxy)[0]
            best = np.argmax(ious)
            if ious[best] > 0.9 and det["class_id"] == labels[best]:
                total_matched += 1
                # 매칭된 박스의 중심/크기 오차 측정
                gx1, gy1, gx2, gy2 = gt_xyxy[best]
                dx1, dy1, dx2, dy2 = det["box"]
                center_err = max(abs((gx1 + gx2) / 2 - (dx1 + dx2) / 2),
                                abs((gy1 + gy2) / 2 - (dy1 + dy2) / 2))
                size_err = max(abs((gx2 - gx1) - (dx2 - dx1)), abs((gy2 - gy1) - (dy2 - dy1)))
                max_center_err = max(max_center_err, center_err)
                max_size_err = max(max_size_err, size_err)

        if len(detections) != len(boxes):
            print(f"[trial {trial}] WARNING: detections={len(detections)} vs gt={len(boxes)}")

    print(f"\nTotal GT boxes: {total_gt}")
    print(f"Matched (IoU>0.9 & correct class): {total_matched}")
    print(f"Max center pixel error: {max_center_err:.4f}")
    print(f"Max size pixel error: {max_size_err:.4f}")
    assert total_matched == total_gt, "encode/decode 라운드트립 불일치!"
    assert max_center_err < 1e-2, "중심 좌표 오차가 너무 큼"
    print("\n[PASS] encode -> decode -> NMS 라운드트립 검증 통과")


if __name__ == "__main__":
    main()
