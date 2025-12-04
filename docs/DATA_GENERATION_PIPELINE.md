# SpatialReasoner Data Generation Pipeline

Multi-View 3D Spatial Reasoning 데이터 생성 파이프라인의 전체 구조를 설명합니다.

## 1. 파이프라인 개요

```
Image (RGB)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Depth Estimation                                    │
│   - Model: Depth Anything v2 (metric depth)                 │
│   - Output: depth map (H x W), intrinsics K (3x3)           │
│   - Range: 0.1m ~ 50m                                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Instance Segmentation                               │
│   - Model: SAM2 (Segment Anything 2)                        │
│   - Output: masks, boxes, areas, labels                     │
│   - Filter: min_area=500 pixels                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: View Synthesis (Optional)                           │
│   - Method: Depth-based point cloud warping                 │
│   - Rotation angles: [-5, 0, +5] degrees                    │
│   - Output: Multi-view images                               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 6-DOF Pose Estimation                               │
│   - Position (x, y, z): Depth-based 3D centroid             │
│   - Quaternion [w, x, y, z]: PCA-based orientation          │
│   - Forward direction: object's +Z axis                     │
│   - Left direction: object's +X axis                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: QA Generation                                       │
│   - 14 category types (matching 3DSRBench)                  │
│   - MCQ format (A, B, C, D options)                         │
│   - Chain-of-Thought (CoT) with REQUIRED_PHRASES patterns   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Training Data JSON
```

---

## 2. 파일 구조

```
src/data_generation/
├── __init__.py
├── config.py              # Configuration dataclasses
├── depth_estimation.py    # Depth Anything v2 wrapper
├── segmentation.py        # SAM2 wrapper
├── view_synthesis.py      # Multi-view generation
├── pose_estimation.py     # 6-DOF pose extraction
├── qa_generation.py       # QA pair + CoT generation
└── pipeline.py            # Full pipeline orchestration

src/spatial_reasoner/utils/
├── quaternion.py          # Quaternion operations
└── geometry.py            # 3D geometry utilities
```

---

## 3. QA Generation 상세

### 3.1 지원 카테고리 (14개)

| Category | Type | REQUIRED_PHRASES |
|----------|------|------------------|
| `height_higher` | Location | 2x location |
| `location_above` | Location | 2x location |
| `location_closer_to_camera` | Location | 2x location, 2x dist_from |
| `location_next_to` | Distance | 2x location, 1x dist_between |
| `orientation_in_front_of` | Orientation | 2x location, vector, front, cosine/angle |
| `orientation_on_the_left` | Orientation | 2x location, vector, left, cosine/angle |
| `orientation_viewpoint` | Orientation | location, vector, front, cosine/angle, left, cosine/angle |
| `multi_object_closer_to` | Multi-object | 3x location, 2x dist_between |
| `multi_object_parallel` | Multi-object | 2x front, cosine/angle |
| `multi_object_same_direction` | Multi-object | 2x front, cosine/angle |
| `multi_object_facing` | Multi-object | 3x location, front, 2x vector, 2x cosine/angle |
| `multi_object_viewpoint_towards_object` | Multi-object | 2x location, vector, front, cosine/angle, left, cosine/angle |
| `rotation_angle_difference` | Rotation | 2x front, rotation_angle/angle |
| `rotation_facing_toward` | Rotation | 2x location, front, vector, cosine/angle |

### 3.2 REQUIRED_PHRASES 패턴

`reward.py`의 `process_reward`가 CoT를 검증하는 정규표현식 패턴:

```python
phrases_dict = {
    "location":    r"3D location of .*? \(-?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+\)",
    "vector":      r"vector from .*? to .*? \(-?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+\)",
    "dist_from":   r"distance from .*? to the camera .*? \d+\.\d+",
    "dist_between": r"distance between .*? \d+\.\d+",
    "cosine":      r"cosine similarity between .*? -?\d+\.\d+",
    "angle":       r"angle between .*? \d+",
    "front":       r"front direction of .*? \(-?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+\)",
    "left":        r"left direction of .*? \(-?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+\)",
    "rotation_angle": r"rotation angle.*?-?\d+\.?\d*\s*degrees?",
}
```

### 3.3 CoT 생성 예시

**Category: `location_next_to`**
```
<think>Let me analyze this spatial reasoning question step by step.
The 3D location of the chair is (1.50, 0.80, 5.20).
The 3D location of the table is (2.30, 0.75, 5.80).
The distance between the chair and the table is 0.96.
Based on this analysis, the answer is 0.96 meters.</think>
<answer>B. 0.96 meters</answer>
```

**Category: `rotation_facing_toward`**
```
<think>Let me analyze this spatial reasoning question step by step.
The 3D location of the car is (3.20, 1.10, 8.50).
The 3D location of the person is (5.60, 0.90, 12.30).
The front direction of the car is (0.85, 0.00, 0.52).
The vector from the car to the person is (0.47, -0.04, 0.88).
The cosine similarity between these vectors is 0.86.
The angle between these vectors is 31 degrees.
Based on this analysis, the answer is Yes.</think>
<answer>A. Yes</answer>
```

---

## 4. 출력 데이터 포맷

### 4.1 QA JSON 구조

```json
{
  "image_filename": "img_001.jpg",
  "question": "Is the chair to the left of the table?",
  "A": "Yes",
  "B": "No",
  "answer_name": "A",
  "answer_cot": "<think>...</think><answer>A. Yes</answer>",
  "category": "orientation_on_the_left",
  "query_type": "directional",
  "query_subtype": "orientation_on_the_left"
}
```

### 4.2 SFT Training과 호환성

`sft.py`에서 기대하는 필드:
- `image_filename`: 이미지 경로
- `question`: 질문
- `A`, `B`, `C`, `D`: MCQ 옵션
- `answer_cot`: `<think>...</think><answer>...</answer>` 포맷
- `category`: REQUIRED_PHRASES 카테고리

### 4.3 GRPO Training과 호환성

`grpo.py`의 `process_reward`가 `answer_cot` 내용에서:
1. `phrases_dict` 패턴 매칭
2. TP (True Positive), FN (False Negative), FP (False Positive) 계산
3. Accuracy = (TP + TN) / (TP + TN + FP + FN)

---

## 5. 실행 방법

### 5.1 데이터 생성

```bash
# Multi-view 데이터 생성
bash local_scripts/generate_multiview_data.sh 1000

# 또는 직접 실행
python scripts/generate_data.py \
    --input_dir ./data/openimages \
    --output_dir ./data/multiview \
    --num_images 1000
```

### 5.2 SFT 학습

```bash
bash local_scripts/spatialreasoner-sft-multiview.sh
```

### 5.3 테스트

```python
python scripts/test_pipeline.py
```

---

## 6. 설정 옵션

### 6.1 DataGenerationConfig

```python
@dataclass
class DataGenerationConfig:
    # Input/Output
    input_dir: str = "./data/openimages"
    output_dir: str = "./data/generated"
    num_images: int = 20000

    # Depth
    depth: DepthConfig = field(default_factory=DepthConfig)

    # Segmentation
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    # View Synthesis
    view_synthesis: ViewSynthesisConfig = field(default_factory=ViewSynthesisConfig)

    # Pose Estimation
    pose_estimation: PoseEstimationConfig = field(default_factory=PoseEstimationConfig)

    # QA Generation
    qa_generation: QAGenerationConfig = field(default_factory=QAGenerationConfig)
```

### 6.2 주요 설정

| Config | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| SegmentationConfig | points_per_side | 32 | SAM2 grid density (8 for faster) |
| ViewSynthesisConfig | rotation_angles | [-5, 0, 5] | View angles in degrees |
| QAGenerationConfig | num_qa_per_image | 10 | QA pairs per image |
| QAGenerationConfig | rotation_ratio | 0.3 | Rotation query ratio |

---

## 7. 성능 벤치마크

| Component | Time (1 image) | GPU Memory |
|-----------|---------------|------------|
| Depth Estimation | ~12s | ~4GB |
| Segmentation | ~10s | ~6GB |
| View Synthesis | ~4s | ~2GB |
| Pose + QA | <1s | CPU only |
| **Total** | **~27s** | **~8GB peak** |

---

## 8. 확장 가능성

1. **Multi-view augmentation**: 더 다양한 각도에서 뷰 생성
2. **Rotation queries**: quaternion 기반 정밀 회전 분석
3. **Temporal reasoning**: 동영상 기반 시간적 공간 추론
4. **3D reconstruction**: NeRF/3DGS 기반 장면 재구성

---

## 문의

이 파이프라인에 대한 질문은 [GitHub Issues](https://github.com/your-repo/SpatialReasoner/issues)를 통해 문의해 주세요.
