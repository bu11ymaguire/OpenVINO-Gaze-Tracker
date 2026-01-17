# Fine-Tuning Guide for OpenVINO (OTX)

이 폴더는 OpenVINO 모델(특히 얼굴 인식 등)을 **한국인/동양인 데이터**로 파인튜닝하기 위한 공간입니다.
OpenVINO Training Extensions (OTX)를 사용합니다.

## 1. 환경 설정 (Installation)

파인튜닝을 위해서는 학습 프레임워크(`otx`) 설치가 필요합니다.
아래 명령어로 설치할 수 있습니다:

```bash
pip install otx
```

## 2. 데이터셋 준비 (Dataset Preparation)

OTX는 보통 **COCO**나 **YOLO**, **VOC** 포맷을 지원합니다.
얼굴 감지(Face Detection)의 경우, 이미지와 얼굴 위치(BBox)가 포함된 데이터가 필요합니다.

**추천 구조 (COCO Format):**
```
tuning/
  └── dataset/
      ├── train/
      │   ├── images/ (001.jpg, 002.jpg ...)
      │   └── annotations.json (COCO format)
      └── val/
          ├── images/
          └── annotations.json
```

## 3. 학습 가능한 모델 템플릿 찾기

설치가 완료되면 아래 명령어로 사용 가능한 얼굴 감지 모델을 찾습니다.

```bash
otx find --template "Face Detection"
```

## 4. 학습 시작 (Training)

```bash
otx train \
  --template <TEMPLATE_ID> \
  --data ./dataset/ \
  --output ./output/
```

## 5. OpenVINO 모델로 내보내기 (Export)

학습된 모델을 다시 `.xml`로 변환합니다.

```bash
otx export \
  --load-weights ./output/weights.pth \
  --output ./exported_model/
```
