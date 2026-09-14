"""
YOLO(v2/v3 스타일) 구현에 필요한 순수 파이썬/넘파이 유틸리티.

이 파일의 함수들은 torch에 의존하지 않습니다. 그래서
- 앵커박스 매칭
- GT -> 학습 타겟 인코딩
- 모델 출력 -> 박스 디코딩
- NMS (Non-Max Suppression)
같은 로직을, PyTorch 없이도 numpy만으로 단독 테스트할 수 있습니다.
(실제 이 저장소에서 torch를 설치할 수 없어 numpy로 먼저 검증했습니다.)
"""
import numpy as np

# -----------------------------
# 전역 설정 (데이터셋/모델과 공유)
# -----------------------------
IMG_SIZE = 128           # 입력 이미지 한 변 크기
S = 8                    # 그리드 크기 (IMG_SIZE / 2^4 downsample)
CLASS_NAMES = ["circle", "square", "triangle"] #탐지할 클래스 이름
NUM_CLASSES = len(CLASS_NAMES) #탐지할 클래스 개수

# 앵커박스: 그리드 셀 단위 (w, h). 합성 데이터셋의 도형 크기(16~56px, cell=16px)에
# 맞춰 대략적으로 고른 값들 (실제 YOLOv2처럼 k-means로 구해도 되지만 데모 목적상 수동 지정).
ANCHORS = np.array([
    [1.0, 1.0],
    [2.0, 2.0],
    [3.0, 1.5],
    [1.5, 3.0],
    [3.5, 3.5],
], dtype=np.float32)
NUM_ANCHORS = len(ANCHORS)


# -----------------------------
# IoU 계산
# -----------------------------
def iou_wh(wh1, wh2): 
    """앵커박스 매칭용: 중심을 (0,0)에 맞췄다고 가정하고 폭/높이만으로 IoU 계산.
    wh1, wh2: (w, h) 튜플/배열
    """
    w1, h1 = wh1
    w2, h2 = wh2
    inter = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-9)


def iou_xyxy(boxes1, boxes2):
    """일반 IoU. boxes1: (N,4), boxes2: (M,4) in (x1,y1,x2,y2). -> (N,M) IoU 행렬"""
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.clip(x2 - x1, a_min=0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0, a_max=None)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-9)


# -----------------------------
# 박스 형식 변환
# -----------------------------
def xywh_to_xyxy(box):
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


# -----------------------------
# GT -> 학습 타겟 인코딩 (YOLOv2 스타일)
# -----------------------------
def encode_targets(boxes, labels, S=S, anchors=ANCHORS, num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    boxes: (N,4) 정규화된 (cx,cy,w,h), 0~1 범위 (이미지 크기로 나눈 값)
    labels: (N,) 클래스 인덱스
    반환: target, shape (S, S, num_anchors, 5+num_classes)
        마지막 축 = [tx, ty, tw, th, obj(1/0), one-hot classes...]
        tx,ty: 셀 내 offset (0~1), tw,th: log-space 앵커 대비 비율
    """
    num_anchors = len(anchors)
    target = np.zeros((S, S, num_anchors, 5 + num_classes), dtype=np.float32)

    for box, label in zip(boxes, labels):
        cx, cy, w, h = box
        # 그리드 셀 좌표
        gx, gy = cx * S, cy * S
        gi, gj = int(gx), int(gy)
        gi = min(max(gi, 0), S - 1)
        gj = min(max(gj, 0), S - 1)

        # 셀 단위 폭/높이 (앵커와 같은 단위로 비교하기 위함)
        gw, gh = w * S, h * S

        # IoU(w,h 기준)가 가장 큰 앵커를 책임 앵커로 선택
        ious = [iou_wh((gw, gh), (aw, ah)) for aw, ah in anchors]
        best_a = int(np.argmax(ious))

        tx = gx - gi
        ty = gy - gj
        tw = np.log(max(gw, 1e-9) / anchors[best_a][0])
        th = np.log(max(gh, 1e-9) / anchors[best_a][1])

        target[gj, gi, best_a, 0] = tx
        target[gj, gi, best_a, 1] = ty
        target[gj, gi, best_a, 2] = tw
        target[gj, gi, best_a, 3] = th
        target[gj, gi, best_a, 4] = 1.0
        target[gj, gi, best_a, 5 + int(label)] = 1.0

    return target


# -----------------------------
# 모델 출력 -> 박스 디코딩
# -----------------------------
def decode_predictions(pred, anchors=ANCHORS, S=S, conf_thresh=0.3, img_size=IMG_SIZE, class_names=None):
    """
    pred: numpy array, shape (S, S, num_anchors, 5+num_classes)
        여기서 tx,ty,obj는 이미 sigmoid가 적용된 값(0~1), tw,th는 raw,
        class는 raw logits(softmax 적용 전)이라고 가정.
    class_names: 클래스 이름 리스트 (기본값은 이 파일의 CLASS_NAMES=도형 3종;
        다른 데이터셋을 쓸 때는 override해서 넘기면 됨, 예: vehicles_config.CLASS_NAMES)
    반환: list of dict {box:(x1,y1,x2,y2) in pixel, conf, class_id, class_name, score}
    """
    if class_names is None:
        class_names = CLASS_NAMES
    num_anchors = len(anchors)
    detections = []
    for gj in range(S):
        for gi in range(S):
            for a in range(num_anchors):
                obj = pred[gj, gi, a, 4]
                if obj < conf_thresh:
                    continue
                tx, ty, tw, th = pred[gj, gi, a, 0:4]
                cx = (gi + tx) / S
                cy = (gj + ty) / S
                w = anchors[a][0] * np.exp(tw) / S
                h = anchors[a][1] * np.exp(th) / S

                class_logits = pred[gj, gi, a, 5:]
                class_probs = _softmax(class_logits)
                class_id = int(np.argmax(class_probs))
                score = float(obj * class_probs[class_id])

                x1, y1, x2, y2 = xywh_to_xyxy((cx, cy, w, h))
                detections.append({
                    "box": (x1 * img_size, y1 * img_size, x2 * img_size, y2 * img_size),
                    "conf": float(obj),
                    "class_id": class_id,
                    "class_name": class_names[class_id] if class_id < len(class_names) else str(class_id),
                    "score": score,
                })
    return detections


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


# -----------------------------
# Non-Max Suppression
# -----------------------------
def non_max_suppression(detections, iou_thresh=0.45):
    """detections: decode_predictions()가 반환한 list of dict. 클래스별로 NMS 수행."""
    if not detections:
        return []

    by_class = {}
    for det in detections:
        by_class.setdefault(det["class_id"], []).append(det)

    kept = []
    for class_id, dets in by_class.items():
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        boxes = np.array([d["box"] for d in dets], dtype=np.float32)
        used = np.zeros(len(dets), dtype=bool)
        for i in range(len(dets)):
            if used[i]:
                continue
            kept.append(dets[i])
            if i + 1 >= len(dets):
                continue
            ious = iou_xyxy(boxes[i:i + 1], boxes[i + 1:])[0]
            for offset, iou_val in enumerate(ious):
                if iou_val > iou_thresh:
                    used[i + 1 + offset] = True
    return kept
