# MVG Final Presentation Scripts

## Hallucination 검증 결과

### ✅ 검증 완료 - 정확한 내용

| 항목 | 스크립트/슬라이드 내용 | 논문 원문 확인 |
|------|----------------------|---------------|
| 3D priors | "depth-aligned 3D priors" | "3D priors that are warped using metric depth and camera poses" ✓ |
| Novel views 수 | "up to 100 novel views" | "generate up to 100 novel views... with a single forward process" ✓ |
| Depth alignment | "using MVS or COLMAP" | "align with SfM", MVS/COLMAP은 SfM 방법론으로 사용됨 ✓ |
| RANSAC | "via RANSAC" | "achieve the rescale and shift coefficient by RANSAC" ✓ |
| Inference 파라미터 | "nframe, cam_traj, d_phi" | GitHub에서 동일하게 확인 ✓ |

### ⚠️ 수정 완료 - Hallucination 또는 부정확한 표현

#### 1. Slide 10 스크립트 → ✅ 수정됨
- 기존의 기술적 세부사항(RANSAC, MVS, COLMAP) 제거
- 핵심 개념(depth-guided geometric consistency)을 두괄식으로 재구성
- 청중이 이해하기 쉬운 추상적 설명으로 변경

#### 2. Slide 7 스크립트 → ✅ 수정됨
- MVGenMaster가 무엇인지 명확히 소개 추가
- "왜 multi-view가 도움이 되는지" 핵심 아이디어를 첫 문장에 배치

---

## 전체 스크립트 (수정 반영)

### Slide 1: Title
> Hello, my name is Seungpil Lee. Today, I will present our course project titled "Enhancing 3D Spatial Reasoning in Vision-Language Models through Multi-View Augmentation." This is joint work with Daeyeon Kim.

**예상 시간:** 15초

---

### Slide 2: Research Goal
> The core challenge in 3D spatial reasoning stems from the inherent ambiguity when inferring three-dimensional orientation from a single two-dimensional image. Current Vision-Language Models achieve below 50% accuracy on orientation-related tasks, which is essentially random guessing.
>
> Our research objective is to address this limitation through multi-view data augmentation. By training models on images rendered from multiple viewpoints, we aim to enable more robust and viewpoint-aware spatial reasoning capabilities.
>
> For this project, Seungpil Lee was responsible for data augmentation pipeline development, model training, evaluation, and this presentation, while Daeyeon Kim contributed to data augmentation and slide preparation.

**예상 시간:** 45초

---

### Slide 3: Vision-Language Model (VLM)
> A Vision-Language Model, or VLM, is a multimodal AI architecture that processes both visual and textual information. The model consists of an image encoder for extracting visual features, a multimodal fusion layer, and a text decoder for generating responses.
>
> Among various VLM applications, we focus specifically on Visual Question Answering, where the model must understand spatial relationships within images to answer questions accurately. This task is particularly challenging because it requires genuine 3D understanding from 2D observations.

**예상 시간:** 30초

---

### Slide 4: Related Work (SpatialReasoner)
> The recently proposed SpatialReasoner attempts to solve this problem through an explicit 3D perception process. Instead of models simply looking at images and reasoning implicitly, it first computes 3D coordinates and vectors, then provides answers based on these calculations. For example, when asked "from the man's perspective, is the water bottle on the left or right," while conventional models make vague guesses, SpatialReasoner calculates the 3D positions of the person and bottle, then provides accurate answers through vector operations.

**예상 시간:** 35초

---

### Slide 5: Problem Statement
> Understanding 3D space from 2D images remains challenging for Vision-Language models. As shown in 3DSRBench, models struggle with questions about height, location, orientation, and multi-object relationships. Especially on rotation benchmarks, like SpinBench, VLM has shown significant weakness, showing accuracy under 50%.

**예상 시간:** 25초

---

### Slide 6: Benchmark Results Table (3DSR)
> The 3DSRBench results reveal that orientation tasks exhibit the lowest performance among all spatial reasoning categories. Looking at the table, while height and location tasks achieve 55 to 68 percent accuracy, orientation tasks drop to around 39 percent.
>
> We specifically target three orientation subtasks. First, orientation_in_front_of determines whether one object is positioned in front of another. Second, orientation_on_the_left assesses lateral positioning from an object's perspective. Third, orientation_viewpoint identifies which side of an object faces the camera. These tasks require understanding of intrinsic object orientations, which is fundamentally ambiguous from single-view observations.

**예상 시간:** 45초

---

### Slide 7: Proposed Method
> Our key idea is simple: if a model sees the same scene from multiple angles during training, it should better understand 3D spatial structure.
>
> To implement this, we use MVGenMaster, a novel view synthesis model that generates what an image would look like from different camera angles. Given a single photo, it can create realistic images as if taken from rotated viewpoints while preserving scene geometry.
>
> Our pipeline has three steps. First, we generate rotated views of training images. Second, we update the spatial labels—direction vectors, 3D bounding boxes, and reasoning chains—to match each new viewpoint. Third, we train on this augmented dataset to improve orientation reasoning.

**예상 시간:** 40초

---

### Slide 8: Pipeline
> The pipeline is as follows. We first augment novel-view images using MVGenMaster. Then, we perform label augmentation in two ways:
> A geometry-based method, where we consider only horizontal rotation and apply a rotation matrix around the Z-axis.
> A model-based method, where we use Qwen2.5-VL to generate a reasoning format and derive labels inferred from the ground truth, which are then used as supervision.

**예상 시간:** 30초

---

### Slide 9: Data Format
> The SpatialReasoner dataset follows a structured format with nine fields. Each sample contains a question about spatial relationships, corresponding answer choices, and chain-of-thought reasoning that explains the solution process.
>
> Crucially for our augmentation approach, three fields contain geometric information that must be transformed when generating novel views. The bounding_box field stores 3D coordinates for each object. The direction field contains front and left direction vectors defining object orientations. The answer_cot field includes numerical computations referencing these spatial quantities. When we rotate the viewpoint, all three fields must be consistently updated to maintain annotation accuracy.

**예상 시간:** 35초

---

### Slide 10: Novel View Image Augmentation (MVGenMaster)
> MVGenMaster solves a key challenge: how do you generate a new viewpoint image that is geometrically consistent with the original?
>
> The solution is to use depth information. The model first estimates the 3D structure of the scene from the input image. Then, when generating a rotated view, it uses this 3D structure to ensure objects appear in the correct positions and scales—not just plausibly, but geometrically accurately.
>
> This depth-guided approach is what distinguishes MVGenMaster from simple image generation: it preserves real spatial relationships rather than hallucinating them.

**예상 시간:** 30초

---

### Slide 11: Novel View Image Augmentation (Inference)
> In practice, we simply provide one image and specify "rotate the camera 10 degrees left" or "30 degrees right." The model then outputs what that scene would look like from the new angle.
>
> For our experiments, we generated views at 10, 20, and 30 degree rotations, creating four versions of each training image.

**예상 시간:** 15초

---

### Slide 12: Spatial Information Editing (HF-Aug)
> This is the first dataset we mentioned earlier. We augmented the labels by applying a rotation matrix, corresponding to the augmentation angle, to the direction vectors and 3D bounding box vectors.

**예상 시간:** 15초

---

### Slide 13: Spatial Information Editing (MVGen-96k) - System Message
> The system message formats the model's reasoning process so that the data can be augmented into the desired form. The think tag becomes the reasoning process, that is, the chain-of-thought label, and the answer tag represents spatial information such as direction and bounding boxes.

**예상 시간:** 20초

---

### Slide 14: Spatial Information Editing (MVGen-96k) - User Message
> We feed the original dataset together with the camera poses of our augmented novel views, and then use them as inputs to run inference with the model.

**예상 시간:** 10초

---

### Slide 15: Spatial Information Editing (MVGen-96k) - Assistant Message
> The model then outputs the next reasoning process as a chain-of-thought, together with the augmented labels, including the updated bounding_box and direction.

**예상 시간:** 15초

---

### Slide 16: Training
> Here, we train the model using two augmented datasets. HF-Aug applies rotation matrices to direction vectors and 3D bounding boxes, generating 2,400 samples. However, the chain-of-thought labels are kept from the original data and not updated to match the rotated values—this creates an inconsistency. MVGen-96k uses Qwen2.5-VL to generate all labels including directions, bounding boxes, and chain-of-thought reasoning, resulting in 96,000 samples across four viewpoints.

**예상 시간:** 25초

---

### Slide 17: Evaluation
> The evaluation results show mixed outcomes across our target tasks.
>
> For HF-Aug, we observe modest improvements on orientation_in_front_of, increasing from 49.7 to 53.1 percent, and orientation_viewpoint, from 16.0 to 18.3 percent. However, orientation_on_the_left decreases from 52.6 to 44.6 percent.
>
> MVGen-96k demonstrates substantial gains on multi-object tasks, particularly multi_object_facing improving from 33.1 to 48.6 percent, and multi_object_same_direction from 32.0 to 42.9 percent. However, it shows degraded performance on location-based tasks.
>
> Overall, these results are insufficient to claim clear performance gains on our primary target tasks. The inconsistent improvements suggest that our augmentation strategy requires refinement to achieve reliable orientation reasoning enhancement.

**예상 시간:** 50초

---

### Slide 18: Results Analysis
> We identified four main factors that may explain the limited performance gains.
>
> First, there is a resolution mismatch between training and evaluation. The model was trained on low-resolution images at 576 by 384 pixels but evaluated on high-resolution images at 1024 by 768 pixels, causing significant loss of fine-grained visual details.
>
> Second, the data distribution is imbalanced. Augmented rotated images dominate the training set, leading to overfitting on rotated low-resolution views and poor generalization to original test images.
>
> Third, for MVGen-96k specifically, there exists a viewpoint-label inconsistency. Although images are generated from different viewpoints, the ground-truth answers remain identical across all views, which may confuse the model during training.
>
> Fourth, for HF-Aug, the chain-of-thought labels were not updated to match the rotated spatial values, creating inconsistency between the CoT reasoning and the actual transformed coordinates.

**예상 시간:** 50초

---

### Slide 19: Limitation, Future Work
> Our approach has several limitations that warrant discussion.
>
> The dataset quality is highly dependent on MVGenMaster's generation quality. In some cases, objects disappear or become distorted in the generated images, introducing noise into the training data.
>
> Additionally, even depth-guided novel view synthesis can contain geometric errors, particularly in regions with complex occlusions or transparent materials.
>
> For future work, we plan to explore incorporating explicit spatial features extracted from depth estimation directly into the model architecture. We also aim to investigate more sophisticated multi-view augmentation strategies that can produce higher-quality, geometrically consistent training datasets.

**예상 시간:** 35초

---

### Slide 20: Thank You
> Thank you. I'm happy to answer questions.

**예상 시간:** 5초

---

## 시간 요약

| 구간 | 슬라이드 | 예상 시간 |
|------|---------|----------|
| Introduction | 1-2 | 1분 00초 |
| Background | 3-6 | 2분 15초 |
| Method | 7-11 | 2분 10초 |
| Data Augmentation | 12-16 | 1분 25초 |
| Results & Discussion | 17-19 | 2분 15초 |
| Closing | 20 | 5초 |
| **Total** | | **약 9분 10초** |

### 8분 목표 달성을 위한 권장 사항
1. Slide 17-18을 약간 빠르게 전달 (핵심 수치만 강조)
2. Slide 9의 데이터 형식 설명 축약 가능
3. 발표 속도를 140-150 wpm으로 조절
