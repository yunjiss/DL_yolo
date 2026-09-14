"""
loss.py의 마스킹/합산 로직을 torch 없이 numpy로 그대로 재현해서
손으로 계산한 값과 일치하는지 확인하는 sanity check.
(torch를 이 샌드박스에 설치할 수 없어, 로직 자체를 numpy로 먼저 검증)
"""
import numpy as np
from utils import S, NUM_ANCHORS, NUM_CLASSES


def mse_sum(a, b):
    return float(np.sum((a - b) ** 2))


def cross_entropy_sum(logits, target_idx):
    # logits: (N,C), target_idx: (N,)
    logits = logits - logits.max(axis=1, keepdims=True)
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
    return float(-np.sum(log_probs[np.arange(len(target_idx)), target_idx]))


def yolo_loss_numpy(pred, target, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_class=1.0):
    obj_mask = target[..., 4] == 1
    noobj_mask = ~obj_mask

    coord_loss = mse_sum(pred[..., 0:4][obj_mask], target[..., 0:4][obj_mask]) if obj_mask.any() else 0.0
    obj_loss = mse_sum(pred[..., 4][obj_mask], target[..., 4][obj_mask]) if obj_mask.any() else 0.0
    noobj_loss = mse_sum(pred[..., 4][noobj_mask], target[..., 4][noobj_mask]) if noobj_mask.any() else 0.0

    if obj_mask.any():
        pred_cls = pred[..., 5:][obj_mask]
        target_cls = target[..., 5:][obj_mask].argmax(axis=-1)
        class_loss = cross_entropy_sum(pred_cls, target_cls)
    else:
        class_loss = 0.0

    B = pred.shape[0]
    total = (lambda_coord * coord_loss + lambda_obj * obj_loss + lambda_noobj * noobj_loss + lambda_class * class_loss) / B
    return total, {
        "total": total, "coord": coord_loss / B, "obj": obj_loss / B,
        "noobj": noobj_loss / B, "class": class_loss / B,
    }


def main():
    # --- 아주 작은 손계산 가능한 예시: B=1, 물체 1개만 있는 케이스 ---
    B, C = 1, NUM_CLASSES
    pred = np.zeros((B, S, S, NUM_ANCHORS, 5 + C), dtype=np.float64)
    target = np.zeros((B, S, S, NUM_ANCHORS, 5 + C), dtype=np.float64)

    # 물체 위치: (gj=2, gi=3, anchor=1), 클래스 0
    target[0, 2, 3, 1, 0:4] = [0.3, 0.7, 0.1, -0.2]  # tx,ty,tw,th
    target[0, 2, 3, 1, 4] = 1.0
    target[0, 2, 3, 1, 5] = 1.0  # class 0 one-hot

    # 예측값 (일부러 정답과 다르게)
    pred[0, 2, 3, 1, 0:4] = [0.2, 0.5, 0.0, 0.0]
    pred[0, 2, 3, 1, 4] = 0.6  # obj confidence
    pred[0, 2, 3, 1, 5:] = [2.0, 0.0, 0.0]  # class logits

    # 나머지 위치(no-object)의 예측 confidence는 전부 0.1로 설정
    pred[..., 4] = 0.1
    pred[0, 2, 3, 1, 4] = 0.6  # 물체 위치는 덮어쓰지 않도록 다시 설정

    total, parts = yolo_loss_numpy(pred, target)

    # --- 손으로 직접 계산 ---
    coord_expected = (0.3 - 0.2) ** 2 + (0.7 - 0.5) ** 2 + (0.1 - 0.0) ** 2 + (-0.2 - 0.0) ** 2
    obj_expected = (0.6 - 1.0) ** 2
    n_noobj = S * S * NUM_ANCHORS - 1
    noobj_expected = n_noobj * (0.1 - 0.0) ** 2
    # cross entropy: logits [2,0,0], target class 0
    logits = np.array([2.0, 0.0, 0.0])
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    class_expected = -np.log(probs[0])

    total_expected = 5.0 * coord_expected + 1.0 * obj_expected + 0.5 * noobj_expected + 1.0 * class_expected

    print("computed:", parts)
    print("expected coord:", coord_expected, "obj:", obj_expected, "noobj:", noobj_expected, "class:", class_expected)
    print("expected total:", total_expected)

    assert abs(parts["coord"] - coord_expected) < 1e-9
    assert abs(parts["obj"] - obj_expected) < 1e-9
    assert abs(parts["noobj"] - noobj_expected) < 1e-9
    assert abs(parts["class"] - class_expected) < 1e-6
    assert abs(total - total_expected) < 1e-6

    print("\n[PASS] YOLO loss 마스킹/합산 로직이 손계산과 일치함")


if __name__ == "__main__":
    main()
