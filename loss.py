"""
YOLOv2 스타일 손실 함수.

model(x)의 출력(pred)과 utils.encode_targets()로 만든 정답(target)을 받아서
아래 네 항의 가중합을 계산한다 (YOLO 논문의 손실 설계를 따름):

    1) coord loss  : 물체가 있는 (grid, anchor) 위치에서 tx,ty,tw,th 좌표 회귀 오차 (SSE)
    2) obj loss     : 물체가 있는 위치에서 confidence(objectness)가 1이 되도록 (SSE)
    3) noobj loss   : 물체가 없는 위치에서 confidence가 0이 되도록 (SSE, 작은 가중치)
    4) class loss   : 물체가 있는 위치에서 클래스 분류 (Cross-Entropy)

pred, target shape: (B, S, S, num_anchors, 5+num_classes)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_class=1.0):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

    def forward(self, pred, target):
        device = pred.device
        obj_mask = target[..., 4] == 1        # (B,S,S,A) bool
        noobj_mask = ~obj_mask
        zero = torch.tensor(0.0, device=device)

        # 1) localization loss
        if obj_mask.any():
            pred_box = pred[..., 0:4][obj_mask]
            target_box = target[..., 0:4][obj_mask]
            coord_loss = F.mse_loss(pred_box, target_box, reduction="sum")
        else:
            coord_loss = zero

        # 2) objectness / 3) no-objectness loss
        pred_obj = pred[..., 4]
        target_obj = target[..., 4]
        obj_loss = (
            F.mse_loss(pred_obj[obj_mask], target_obj[obj_mask], reduction="sum")
            if obj_mask.any() else zero
        )
        noobj_loss = (
            F.mse_loss(pred_obj[noobj_mask], target_obj[noobj_mask], reduction="sum")
            if noobj_mask.any() else zero
        )

        # 4) classification loss
        if obj_mask.any():
            pred_cls = pred[..., 5:][obj_mask]                       # (N_obj, C) raw logits
            target_cls = target[..., 5:][obj_mask].argmax(dim=-1)    # (N_obj,)
            class_loss = F.cross_entropy(pred_cls, target_cls, reduction="sum")
        else:
            class_loss = zero

        batch_size = pred.shape[0]
        total = (
            self.lambda_coord * coord_loss
            + self.lambda_obj * obj_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_class * class_loss
        ) / batch_size

        loss_dict = {
            "total": total.item(),
            "coord": coord_loss.item() / batch_size,
            "obj": obj_loss.item() / batch_size,
            "noobj": noobj_loss.item() / batch_size,
            "class": class_loss.item() / batch_size,
        }
        return total, loss_dict


if __name__ == "__main__":
    # 간단한 shape/역전파 sanity check (torch 설치 환경에서 실행)
    from utils import S, NUM_ANCHORS, NUM_CLASSES

    B = 2
    pred = torch.randn(B, S, S, NUM_ANCHORS, 5 + NUM_CLASSES, requires_grad=True)
    target = torch.zeros(B, S, S, NUM_ANCHORS, 5 + NUM_CLASSES)
    target[0, 3, 3, 1, 4] = 1.0
    target[0, 3, 3, 1, 5] = 1.0  # class 0
    target[1, 5, 2, 0, 4] = 1.0
    target[1, 5, 2, 0, 6] = 1.0  # class 1

    criterion = YoloLoss()
    loss, loss_dict = criterion(pred, target)
    print(loss_dict)
    loss.backward()
    print("grad ok:", pred.grad is not None)
