"""
Roboflow 'vehicles.v2-release.yolov8' 데이터셋에 맞춘 설정값.
(다운로드 경로: Universe > roboflow-100 > vehicles-q0x2v > v2, YOLOv8 포맷)

- 원본 이미지 640x480, 이미지당 평균 약 12개 객체(교통 카메라 영상이라 객체가 많고 작음)
- IMG_SIZE=128(도형 데모용 utils.py 기본값)로는 객체가 너무 작아져서 거의 안 보이므로
  416으로 키움. 모델(backbone)은 4번의 stride-2 maxpool로 다운샘플하므로
  S = IMG_SIZE / 16 = 26
- ANCHORS는 실제 이 데이터셋의 train/labels 전체(31,905개 박스)에 대해 IoU 기반
  k-means(k=5)로 직접 계산한 값 (평균 best-IoU ≈ 0.755)
"""
import numpy as np

IMG_SIZE = 416
DOWNSAMPLE = 16          # model.py 백본의 다운샘플 배율 (고정)
S = IMG_SIZE // DOWNSAMPLE  # = 26

CLASS_NAMES = [
    "big bus", "big truck", "bus-l-", "bus-s-", "car", "mid truck",
    "small bus", "small truck", "truck-l-", "truck-m-", "truck-s-", "truck-xl-",
]  # data.yaml의 순서와 반드시 동일해야 함 (라벨 class_id가 이 순서를 그대로 씀)
NUM_CLASSES = len(CLASS_NAMES)  # 12

# train/labels 전체를 IoU 기반 k-means(k=5)로 클러스터링해서 얻은 앵커
# (그리드 셀 단위, S=26 기준). w,h를 오름차순 면적으로 정렬.
ANCHORS = np.array([
    [0.860, 1.104],
    [1.378, 1.902],
    [2.142, 3.045],
    [3.341, 4.691],
    [5.251, 8.490],
], dtype=np.float32)
NUM_ANCHORS = len(ANCHORS)
