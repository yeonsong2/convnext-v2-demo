# ConvNeXt V2 Experiments

ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders ([논문](https://arxiv.org/abs/2301.00808) / [원본 레포](https://github.com/facebookresearch/ConvNeXt-V2))

## 실험 구성

| 실험 | 설명 |
|------|------|
| ImageNet Classification | ConvNeXt V2-Huge 논문 수치 재현 |
| V1 vs V2 비교 | 동일 크기 모델 성능 및 처리 속도 비교 |
| Demo | 커스텀 이미지 추론 |

## 주요 결과

### ConvNeXt V2-Huge ImageNet-1K 평가
| 지표 | 재현 결과 | 논문 수치 |
|------|-----------|-----------|
| Top-1 Accuracy | **86.13%** | 86.3% |
| Top-5 Accuracy | **97.76%** | - |

### ConvNeXt V1 vs V2 비교 (Large 기준)
| 지표 | V1 | V2 | 개선 |
|------|----|----|------|
| Top-1 Accuracy | 84.17% | **85.56%** | **+1.39%** |
| Top-5 Accuracy | 96.84% | 97.55% | +0.71% |
| 파라미터 수 | 197.8M | 198.0M | +0.2M |
| 처리 속도 | 203 img/s | 154 img/s | -50 img/s |

## 핵심 기여 (논문 요약)

### 1. FCMAE (Fully Convolutional Masked Autoencoder)
- 자기지도학습 사전학습 프레임워크
- 기존 MAE를 ConvNet에 맞게 재설계

### 2. GRN (Global Response Normalization)
- V1의 Layer Scale을 대체하는 새로운 정규화 레이어
- 채널 간 feature 경쟁을 강화하여 표현력 향상

```
V1 Block: DWConv → LN → MLP → LayerScale
V2 Block: DWConv → LN → MLP → GRN        ← 핵심 차이
```

## 설치

```bash
pip install torch torchvision
```

## 사용법

### 1. 체크포인트 다운로드
```bash
# V2 Huge만
python download_checkpoints.py --model convnextv2_huge_1k_224 --version v2

# V1, V2 전체
python download_checkpoints.py --version both
```

### 2. ImageNet 평가
```bash
python eval_imagenet.py \
  --model convnextv2_huge \
  --checkpoint checkpoints/convnextv2_huge_1k_224.pt \
  --data_path /path/to/imagenet \
  --gpu 0
```

### 3. V1 vs V2 비교
```bash
python compare_v1_v2.py \
  --size large \
  --data_path /path/to/imagenet \
  --gpu 0
```

### 4. 데모
```bash
# 단일 이미지
python demo.py \
  --image path/to/image.jpg \
  --checkpoint checkpoints/convnextv2_large_1k_224.pt \
  --gpu 0

# V1 vs V2 나란히 비교
python demo.py \
  --image path/to/image.jpg \
  --checkpoint checkpoints/convnextv2_large_1k_224.pt \
  --v1_checkpoint checkpoints/convnext_large_1k_224.pt \
  --compare \
  --gpu 0
```

## 파일 구조

```
ConvNeXt-V2/
├── models/
│   ├── convnextv2.py      # ConvNeXt V2 (GRN 포함)
│   ├── convnextv1.py      # ConvNeXt V1 (비교용)
│   └── utils.py           # LayerNorm, GRN, DropPath
├── results/               # 실험 결과
├── download_checkpoints.py
├── eval_imagenet.py
├── compare_v1_v2.py
└── demo.py
```

## 환경

- Python 3.10
- PyTorch 2.1.0
- NVIDIA RTX A6000 (48GB) × 1
