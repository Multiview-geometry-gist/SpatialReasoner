# SpatialReasoner 학습 방법 비교 분석

## 1. 성능 비교표

### 1.1 전체 성능 (3DSRBench)

| 모델 | Per-Sample | CircularEval | 학습 방식 | 데이터 수 |
|------|------------|--------------|----------|-----------|
| **SFT-v2** | 58.07% | 44.27% | From Scratch | 24,000 |
| **HF-Augmented** | **60.19%** | **45.22%** | Fine-tuning (SFT-v2) | 2,400 |
| **MVGen-96k** | 50.30% | 37.98% | From Scratch | 96,000 |

### 1.2 성능순 정렬

| 순위 | 모델 | CircularEval | Per-Sample | vs SFT-v2 |
|------|------|--------------|------------|-----------|
| 1 | **HF-Augmented** | **45.22%** | 60.19% | **+0.95%** |
| 2 | SFT-v2 | 44.27% | 58.07% | baseline |
| 3 | MVGen-96k | 37.98% | 50.30% | **-6.29%** |

### 1.3 카테고리별 CircularEval 성능 (%)

| 카테고리 | SFT-v2 | HF-Aug | MVGen-96k | Best |
|----------|--------|--------|-----------|------|
| location_closer_to_camera | 70.0 | **72.6** | 58.3 | HF-Aug |
| location_next_to | 63.4 | **66.3** | 49.7 | HF-Aug |
| orientation_on_the_left | **52.6** | 44.6 | 29.7 | SFT-v2 |
| orientation_in_front_of | 49.7 | **53.1** | 50.3 | HF-Aug |
| location_above | 49.1 | **51.1** | 48.6 | HF-Aug |
| multi_object_facing | 33.1 | 37.1 | **48.6** | MVGen |
| multi_object_closer_to | **47.4** | 46.3 | 22.3 | SFT-v2 |
| multi_object_same_direction | 32.0 | 29.1 | **42.9** | MVGen |
| height_higher | 40.0 | **43.1** | 29.4 | HF-Aug |
| multi_object_parallel | **32.6** | **32.6** | 32.0 | Tie |
| multi_object_viewpoint_towards_object | **18.9** | 17.1 | 2.3 | SFT-v2 |
| orientation_viewpoint | 16.0 | 18.3 | **19.4** | MVGen |

---

## 2. 학습 방식 비교: Fine-tuning vs From Scratch

### 2.1 비교 요약

| 항목 | Fine-tuning (HF-Augmented) | From Scratch (MVGen-96k) |
|------|---------------------------|-------------------------|
| **베이스 모델** | SFT-v2 (58% 성능) | Qwen2.5-VL-7B-Instruct |
| **학습 데이터** | 2,400개 | 96,000개 |
| **학습 시간** | ~1시간 | ~3.5시간 |
| **최종 성능** | **45.22%** | 37.98% |
| **성능 변화** | **+0.95%** | -6.29% |
| **효율성** | 매우 높음 | 낮음 |

### 2.2 분석

```
Fine-tuning (HF-Augmented):
├── 장점
│   ├── 기존 지식 보존 (SFT-v2의 58% 성능 기반)
│   ├── 적은 데이터로 효과적 학습 (2,400개)
│   ├── 낮은 learning rate (2e-6)로 안정적 학습
│   └── 원본 이미지 사용으로 품질 유지
└── 결과: +0.95% 성능 향상

From Scratch (MVGen-96k):
├── 문제점
│   ├── 기존 지식 없이 처음부터 학습
│   ├── 저해상도 이미지 (~218k pixels)
│   ├── 시점-정답 불일치로 혼란
│   └── 데이터 불균형 (원본 25%, 증강 75%)
└── 결과: -6.29% 성능 하락
```

---

## 3. 하이퍼파라미터 비교

### 3.1 핵심 하이퍼파라미터

| 파라미터 | SFT-v2 | HF-Augmented | MVGen-96k |
|----------|--------|--------------|-----------|
| **Base Model** | Qwen2.5-VL-7B | SFT-v2 checkpoint | Qwen2.5-VL-7B |
| **Learning Rate** | 5e-6 | 2e-6 | 2e-6 |
| **Epochs** | 10 | 3 | 1 |
| **Batch Size** | 4 | 1 | 1 |
| **Grad Accum Steps** | 1 | 8 | 8 |
| **Effective Batch** | 4 | 8 | 8 |
| **Warmup Ratio** | 0.1 | 0.15 | 0.1 |
| **LR Scheduler** | cosine | cosine | cosine |
| **Max Length** | 2048 | 2048 | 2048 |

### 3.2 이미지 해상도 설정

| 파라미터 | SFT-v2 | HF-Augmented | MVGen-96k |
|----------|--------|--------------|-----------|
| **min_pixels** | 100,352 | 59,716 | 59,716 |
| **max_pixels** | 100,352 | 100,352 | 200,704 |
| **해상도 방식** | 고정 | 가변 | 가변 |
| **실제 학습 이미지** | ~786k px | ~786k px | ~218k px |
| **테스트 이미지** | ~786k px | ~786k px | ~786k px |

### 3.3 학습 규모

| 항목 | SFT-v2 | HF-Augmented | MVGen-96k |
|------|--------|--------------|-----------|
| **학습 샘플 수** | 24,000 | 2,400 | 96,000 |
| **총 학습 스텝** | ~7,000 | ~900 | ~3,000 |
| **GPU 사용** | 4× A100 | 4× A100 | 4× A100 |
| **예상 학습 시간** | ~8시간 | ~1시간 | ~3.5시간 |

---

## 4. 데이터셋 비교

### 4.1 데이터 구성

| 항목 | SFT-v2 | HF-Augmented | MVGen-96k |
|------|--------|--------------|-----------|
| **소스** | ccvl/SpatialReasonerTrain-SFT | HuggingFace RL datasets | MVGenMaster 생성 |
| **이미지 타입** | 원본 OpenImages | +10°/-10° 회전 생성 이미지 | MVGen 생성 |
| **이미지 해상도** | 1024×768 | 576×320 | 576×384 |
| **CoT 형식** | 원본 | 회전 변환 (+10°, -10°) | 시점별 변환 |
| **정답 변화** | - | 동일 | 동일 |

### 4.2 데이터 증강 방식

```
SFT-v2 (원본):
└── 원본 이미지 + 원본 CoT

HF-Augmented (이미지 + CoT 변환):
├── +10° 회전 생성 이미지 + 회전된 CoT 좌표 (1,200개)
└── -10° 회전 생성 이미지 + 회전된 CoT 좌표 (1,200개)
    → 이미지와 CoT 좌표 모두 회전 변환, 정답은 동일

MVGen-96k (이미지 생성):
├── 0° 원본 view (24,000개)
├── 10° 생성 view (24,000개)
├── 20° 생성 view (24,000개)
└── 30° 생성 view (24,000개)
    → 이미지도 변환, CoT도 변환, 정답은 동일
```

---

## 5. 프롬프트 형식

### 5.1 입력 프롬프트 (공통)

```
Consider the real-world 3D locations and orientations of the objects.
Which side of the {object} is facing the camera?
A. front
B. back
C. left
D. right
```

### 5.2 출력 형식

**SFT-v2 / HF-Augmented (CoT 형식)**:
```
The location of the child is (-1.5, 0.8, 13.1).
The vector from the child to camera is hence (1.5, -0.8, -13.1).
The left direction of the child is (0.0, 0.1, -1.0).
The cosine similarity between the vector pointing to camera and
the left direction is 0.98, corresponding to an angle of 11.77 degrees.
...
Therefore, the final answer is C. left.
```

**MVGen-96k (단답형)**:
```
C. left
```

---

## 6. 성능 저하 원인 분석 (MVGen-96k)

### 6.1 핵심 원인

| 순위 | 원인 | 심각도 | 설명 |
|------|------|--------|------|
| 1 | 이미지 해상도 | ★★★★★ | 학습 218k vs 테스트 744k (3.4배 차이) |
| 2 | 시점-정답 불일치 | ★★★★☆ | 이미지는 변하지만 정답은 동일 |
| 3 | 데이터 불균형 | ★★★★☆ | 원본 25% vs 증강 75% |
| 4 | From Scratch | ★★★☆☆ | 기존 지식 없이 처음부터 학습 |

### 6.2 카테고리별 영향

```
가장 큰 성능 하락:
• multi_object_closer_to: -25.1% (47.4% → 22.3%)
• orientation_on_the_left: -22.9% (52.6% → 29.7%)
• multi_object_viewpoint_towards_object: -16.6% (18.9% → 2.3%)

성능 향상 (예외):
• multi_object_facing: +15.4% (33.1% → 48.6%)
• multi_object_same_direction: +10.9% (32.0% → 42.9%)
```

---

## 7. 결론 및 권장사항

### 7.1 방법별 평가

| 방법 | 효율성 | 성능 | 권장 |
|------|--------|------|------|
| **SFT-v2** | 보통 | 44.27% | 베이스라인 |
| **HF-Augmented** | **최고** | **45.22%** | ✅ **권장** |
| **MVGen-96k** | 낮음 | 37.98% | ❌ 수정 필요 |

### 7.2 핵심 인사이트

```
1. Fine-tuning > From Scratch
   - 적은 데이터(2.4k)로 더 좋은 성능
   - 기존 지식 보존이 중요

2. 이미지 품질 > 데이터 양
   - 96k 저해상도 < 2.4k 고해상도
   - 해상도 일치가 필수

3. 데이터 증강 전략
   - 이미지 생성보다 CoT 변환이 효과적
   - 시점-정답 일관성 유지 필요
```

### 7.3 권장 학습 설정

```yaml
# 최적 설정 (HF-Augmented 기반)
model_name_or_path: SFT-v2 checkpoint  # Fine-tuning
learning_rate: 2.0e-06                  # 낮은 LR
num_train_epochs: 3                     # 적은 epoch
gradient_accumulation_steps: 8          # 큰 effective batch
min_pixels: 100352                      # 고해상도 유지
max_pixels: 100352                      # 고정 해상도
warmup_ratio: 0.15                      # 충분한 warmup
```

### 7.4 향후 개선 방향

1. **MVGen 해상도 향상**: 1024×768로 생성
2. **Mixed Training**: 원본 + MVGen 혼합 (50:50)
3. **시점별 정답 검증**: 시점 불변 질문만 증강
4. **SFT-v2 기반 Fine-tuning**: From Scratch 대신 Fine-tuning
