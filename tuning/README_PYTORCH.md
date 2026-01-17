
# PyTorch Fine-Tuning Guide (Lightweight)

`otx` 설치 문제로 인해, 더 가볍고 확실한 **Pure PyTorch** 방식으로 진행합니다.
우리는 대중적인 **MobileNetV2-SSD** 모델을 사용하여 얼굴 인식을 재학습할 것입니다.

## 1. 데이터셋 구조 준비 (VOC Format)
가장 널리 쓰이는 Pascal VOC 포맷을 따릅니다.

```
tuning/
  └── data/
      ├── images/ (001.jpg, ...)
      └── annotations/ (001.xml, ...)
```
**LabelImg** 같은 툴로 얼굴에 박스를 치고 저장하면 `.xml` 파일이 생깁니다.

## 2. 학습 스크립트 실행
`tuning/train.py`를 실행하여 학습을 시작합니다.
이 스크립트는 `torchvision`의 미리 학습된 SSD 모델을 불러와서 내 데이터에 맞게 재학습합니다.

```bash
python tuning/train.py
```

## 3. OpenVINO 변환
학습이 끝나면 `best_model.pth` 파일이 생성됩니다.
이를 OpenVINO용으로 변환합니다.

```bash
# 1단계: ONNX 변환
python tuning/export_onnx.py

# 2단계: IR 변환
mo --input_model tuning/output/model.onnx --output_dir tuning/output/ir
```
