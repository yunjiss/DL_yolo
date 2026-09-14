"""
YOLOv2/v3 스타일의 작은(tiny) 탐지 모델.

구조:
  입력 (B,3,128,128)
   -> Darknet 스타일 백본 (conv+BN+LeakyReLU, 4번의 stride-2 maxpool로 다운샘플)
   -> feature map (B,256,8,8)   (S=8 그리드)
   -> 1x1 conv 헤드 -> (B, S, S, num_anchors, 5+num_classes)

출력 마지막 축 = [tx, ty, tw, th, obj, class_logits...]
  - tx, ty, obj : forward()에서 sigmoid 적용됨 (0~1)
  - tw, th      : raw (log-space, 앵커 대비 배율에 log 취한 값)
  - class_logits: raw logits (softmax는 loss/decode 단계에서 적용)
"""
import torch
import torch.nn as nn

from utils import NUM_ANCHORS, NUM_CLASSES


def conv_bn_leaky(in_ch, out_ch, kernel_size=3, stride=1):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class YoloBackbone(nn.Module):
    """128x128 입력 -> 8x8 feature map (stride 16), 작은 Darknet 스타일."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_leaky(3, 16),
            nn.MaxPool2d(2, 2),   # 128 -> 64

            conv_bn_leaky(16, 32),
            nn.MaxPool2d(2, 2),   # 64 -> 32

            conv_bn_leaky(32, 64),
            nn.MaxPool2d(2, 2),   # 32 -> 16

            conv_bn_leaky(64, 128),
            nn.MaxPool2d(2, 2),   # 16 -> 8

            conv_bn_leaky(128, 256),
            conv_bn_leaky(256, 256),
        )

    def forward(self, x):
        return self.features(x)


class YoloHead(nn.Module):
    def __init__(self, in_ch=256, num_anchors=NUM_ANCHORS, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        out_ch = num_anchors * (5 + num_classes)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        B, _, Sh, Sw = x.shape
        out = self.conv(x)                                   # (B, A*(5+C), S, S)
        out = out.view(B, self.num_anchors, 5 + self.num_classes, Sh, Sw)
        out = out.permute(0, 3, 4, 1, 2).contiguous()         # (B, S, S, A, 5+C)
        return out


class YoloV2Tiny(nn.Module):
    def __init__(self, num_anchors=NUM_ANCHORS, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = YoloBackbone()
        self.head = YoloHead(256, num_anchors, num_classes)

    def forward(self, x, apply_activation=True):
        feat = self.backbone(x)
        raw = self.head(feat)  # (B,S,S,A,5+C) raw logits

        if not apply_activation:
            return raw

        tx = torch.sigmoid(raw[..., 0:1])
        ty = torch.sigmoid(raw[..., 1:2])
        tw = raw[..., 2:3]
        th = raw[..., 3:4]
        obj = torch.sigmoid(raw[..., 4:5])
        cls = raw[..., 5:]  # raw logits, softmax는 loss/decode에서

        out = torch.cat([tx, ty, tw, th, obj, cls], dim=-1)
        return out


if __name__ == "__main__":
    # 모델이 정상적으로 정의되는지, 출력 shape이 기대대로 나오는지 확인하는 용도.
    # (이 스크립트는 torch가 설치된 환경에서 실행해야 합니다.)
    from utils import IMG_SIZE, S

    model = YoloV2Tiny()
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    out = model(dummy)
    print("output shape:", out.shape)  # 기대: (2, S, S, NUM_ANCHORS, 5+NUM_CLASSES)
    assert out.shape == (2, S, S, NUM_ANCHORS, 5 + NUM_CLASSES)
    print("OK")
