# SCDMN-Sliced Regression 확장 실험: 결과 요약

## TL;DR

CIFAR-10-C 기반 pseudo steering regression 실험에서 **SCDMN-Sliced는 Single 대비 이득을 보이지 못했다.** 모든 configuration에서 Sliced MAE ≥ Single MAE였고, 가장 어려운 context(gaussian_noise)에서조차 유의미한 차이가 없었다. 이는 실험 실패가 아니라 **CIFAR-10-C로는 mode averaging 가설을 검증할 수 없다**는 발견이다. ResNet18의 capacity가 4개 context 동시 학습에 충분해서, single 모델이 애초에 mode averaging 피해를 입지 않는다.

---

## 배경

기존 SCDMN-Sliced는 CIFAR-10-C 분류에서 mask IoU 0.6~0.8로 과공유되어 mode averaging 해소 효과가 드러나지 않았다. "연속값 regression으로 바꾸면 단일 모델의 averaging이 더 직접적으로 드러날 것"이라는 가설로 DriveMoE 스타일의 소극적 검증(Single vs SCDMN MAE 비교)을 세팅했다.

---

## 실험 세팅

### 데이터
- CIFAR-10 + CIFAR-10-C (clean / brightness / frost / gaussian_noise)
- Pseudo steering target: pretrained ResNet18(ImageNet) → 512-d feature → PCA 1D → z-score → tanh(z/2) ∈ [-1, 1]
- src_idx에만 의존 (4 context가 동일 이미지 → 동일 target)
- Target std ≈ 0.437 (constant-prediction MAE 상한 ≈ 0.35)

### 모델
- **Single**: ResNet18-CIFAR(num_classes=1) + tanh, 혼합 context 학습. 21.28M params, 2.32 GFLOPs
- **SCDMN-Sliced**: 기존 구조 + fc(→1) + tanh. sparsity 0.5 기준 post-freeze 0.61 GFLOPs (~26% of Single)

### 학습
- Loss: SmoothL1Loss (beta=0.1)
- Optimizer: SGD + Nesterov, Cosine LR
- Batch: 128, train 10k~30k per context, 30~40 epochs
- Sliced: soft mask warmup → freeze → hard sliced forward

---

## 실험 결과

### 1) Severity 3 (비공식 synthetic corruption)
sparsity 0.5, freeze 5, train 10k, 30 epoch

| | Single | Sliced |
|---|---|---|
| overall MAE | 0.133 | 0.132 |

거의 동일. Corruption 강도가 약해서 context 구분 자체가 의미 없었음.

### 2) Severity 5 (공식 CIFAR-10-C)
sparsity 0.5, freeze 5, train 10k, 30 epoch

| context | Single | Sliced |
|---|---|---|
| clean | 0.178 | 0.245 |
| brightness | 0.358 | 0.388 |
| frost | 0.356 | 0.377 |
| gaussian_noise | 0.388 | 0.384 |
| **overall** | **0.320** | **0.349** |

gaussian_noise에서만 Sliced가 미세하게 앞섰으나 유의미한 차이 아님. 어려운 context일수록 gap이 줄어드는 경향은 있으나 mode averaging 주장하기엔 역부족.

### 3) Severity 5 + 장기 warmup (train 30k, freeze 10)
Sliced 성능 오히려 악화. 손상 context들이 constant-prediction bound(~0.38)에 묶임.

### 4) Severity 3 official + sparsity 0.75 + freeze 8, 40 epochs, train 10k

| | Single | Sliced |
|---|---|---|
| overall MAE | 0.302 (best 0.281) | 0.399 (best 0.371) |

- Sliced clean이 epoch마다 0.38~0.70 사이로 크게 진동
- 손상 context 전부 ~0.38 바닥에서 못 벗어남
- 가장 gap이 벌어진 케이스 (의도한 방향의 반대)

---

## 핵심 관찰

1. **어떤 config에서도 Sliced가 Single을 이기지 못함.** 하드 context조차 미미한 차이.

2. **Warmup이 길수록 Sliced가 나빠짐.** Soft mask warmup 구간에는 거의 학습 신호가 안 들어오고, freeze 시점에 갑자기 학습이 시작되면 남은 epoch이 짧아 수렴 실패.

3. **Sparsity를 키워도(채널 더 많이 써도) 악화.** 문제는 용량이 아니라 학습 역학.

4. **Mode averaging 가설이 CIFAR-10-C에서는 drive되지 않음.** Pseudo target이 src_idx에만 의존해도 Single이 context별 다른 입력으로부터 같은 target을 충분히 뽑아냄. ResNet18 capacity + tanh saturation이 target 분포를 완만하게 만들어 모드가 잘 안 갈라짐.

---

## 해석

CIFAR 실험의 분류와 regression 결과를 종합하면 같은 결론에 도달한다:

> **CIFAR-10-C는 mode averaging을 드러내기에 너무 쉬운 데이터다.**

Mode averaging은 모델 capacity 대비 domain diversity가 클 때 발생하는 현상인데, CIFAR에서는 그 조건이 만들어지지 않는다. ResNet18이 4개 context를 동시에 커버할 capacity가 있어 single 모델이 애초에 averaging 피해를 입지 않는다.

이는 실험 실패가 아니라 **"이 가설을 검증하려면 더 challenging한 데이터가 필요하다"** 는 명확한 다음 단계의 동기가 된다.

---

## 결론 및 다음 단계

### CIFAR 단계는 종료
- 분류: SCDMN 구조가 작동 (gate 패턴, 부분적 specialization 확인)
- Regression: mode averaging이 CIFAR에서 관측되지 않음 (IoU 과공유 관측과 일관)
- 두 결과 모두 supplementary / analysis 섹션 재료로 보존

### 다음 단계: 실제 driving 데이터
Capacity-domain diversity 조건을 만족하는 데이터셋으로 이동.

**comma2k19 서브셋 선택**:
- 실제 도로 주행 영상 + 조향각 레이블
- 다양한 날씨/시간대 포함 (캘리포니아, 2019)
- 서브셋(chunk 1~2개, ~10GB)으로 시작 가능
- 밝기 기반 context 자동 분류 (day_clear / day_overcast / night)

기존 구현 자산(`data/multi_context_cifar_reg.py`, `models/scdmn_sliced_reg.py`, `experiments/trainer_reg.py`)은 재사용 가능하며, 데이터 로더와 context 분류 스크립트만 새로 추가하면 된다.

---

## 가진 자산 (보존)

- `data/multi_context_cifar_reg.py`
- `models/scdmn_sliced_reg.py`
- `models/resnet_baseline_reg.py`
- `experiments/trainer_reg.py`
- `experiments/run_regression.py`
- `scripts/make_pseudo_targets.py`

기존 분류 경로는 건드리지 않음 (재현성 보존).