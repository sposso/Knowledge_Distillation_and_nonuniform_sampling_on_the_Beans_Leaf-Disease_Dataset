# Models used in this distillation experiment

Knowledge distillation: a fine-tuned **Vision Transformer (ViT-Base/16)** teacher
transfers knowledge to a from-scratch **MobileNetV2** student on the
[`beans`](https://huggingface.co/datasets/AI-Lab-Makerere/beans)
3-class leaf-disease dataset (1,034 train / 133 val / 128 test, classes:
`angular_leaf_spot`, `bean_rust`, `healthy`).

---

## Teacher: `merve/beans-vit-224`

[Hugging Face Hub link](https://huggingface.co/merve/beans-vit-224) · License: Apache-2.0

### Architecture
- **Family:** ViT (Vision Transformer) — `ViTForImageClassification`
- **Backbone:** `google/vit-base-patch16-224-in21k` (ViT-Base/16)
- **Input image size:** 224 × 224
- **Patch size:** 16 × 16  → 14 × 14 = 196 image tokens (+ 1 [CLS])
- **Hidden size:** 768
- **Encoder layers:** 12
- **Attention heads:** 12 (head dim 64)
- **MLP intermediate size:** 3,072
- **Activation:** GELU
- **Classification head:** linear `768 → 3` over the [CLS] token

### Parameters
- **Total:** **85,800,963** (~85.8 M)
- **Trainable in this run:** all (the teacher is frozen during distillation in our code, but contains 85.8 M params total)

### Pre-training
- **Source weights:** `google/vit-base-patch16-224-in21k`, pre-trained
  *self-supervised* on **ImageNet-21k** (~14 M images, 21,841 classes), then
  released by Google as a feature backbone (no classification head).
- ViT paper: *An Image Is Worth 16×16 Words* — Dosovitskiy et al., ICLR 2021
  ([arXiv:2010.11929](https://arxiv.org/abs/2010.11929)).

### Fine-tuning on beans (by `merve`)
| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999, ε=1e-8) |
| Learning rate | 5e-5 |
| LR schedule | Linear, warmup ratio 0.1 |
| Train batch size (per device) | 16 |
| Gradient accumulation | 4 |
| **Effective batch size** | **64** |
| Eval batch size | 16 |
| Epochs | 3 |
| Seed | 42 |
| Frameworks | transformers 4.34.0, torch 2.0.1+cu118, datasets 2.14.5 |

### Reported validation results (from the model card)
| Epoch | Step | Val loss | Val accuracy |
|---:|---:|---:|---:|
| 1.00 | 16 | 0.6540 | 0.8828 |
| 1.97 | 32 | 0.4180 | 0.9297 |
| 2.95 | 48 | 0.3256 | 0.9375 |

Test-set accuracy reported on the model card: **0.938** — and reproduced in
this repo (`results/results.md`): **0.9375 (macro-F1 0.9379)** when run in fp16
inference.

---

## Student: `MobileNetV2ForImageClassification` (from scratch)

Built from `MobileNetV2Config()` defaults (no pre-trained weights loaded), with
the classification head sized to 3 classes for beans.

### Architecture
- **Family:** MobileNetV2 (Sandler et al., 2018)
  — `MobileNetV2ForImageClassification`
- **Input image size:** 224 × 224
- **Width multiplier (α):** 1.0
- **Inverted-residual expansion ratio:** 6
- **Depth multiplier:** 1.0
- **Building block:** Inverted residuals with linear bottlenecks
  (1×1 expand → 3×3 depthwise → 1×1 project, with residual when stride=1 and
  in/out channels match).
- **Classification head:** GAP → 1×1 conv → linear `1280 → 3`
- **Activation:** ReLU6

### Parameters
- **Total:** **2,227,715** (~2.23 M) — about **38× smaller** than the teacher.
- **Trainable:** all 2.23 M (random init, full training).

### Pre-training
- **None.** The student is initialized randomly and learns purely from
  - the supervised cross-entropy loss on beans labels, and
  - the KL-divergence to the teacher's softened logits (distillation signal).
- MobileNetV2 paper: *MobileNetV2: Inverted Residuals and Linear Bottlenecks* —
  Sandler et al., CVPR 2018 ([arXiv:1801.04381](https://arxiv.org/abs/1801.04381)).
- (For reference, Google's pretrained `google/mobilenet_v2_1.0_224` on
  ImageNet-1k has ~3.5 M params — slightly more because that release uses a
  1001-way head and the `featureExtractor` head sized for ImageNet rather than
  3 classes.)

### Distillation loss used in this repo
$$\mathcal{L} = (1-\lambda)\,\text{CE}(s,y) + \lambda \, T^2 \, \text{KL}\big(\sigma(s/T)\,\|\,\sigma(t/T)\big)$$

with student logits $s$, teacher logits $t$, true labels $y$, temperature $T$,
and weight $\lambda = 0.5$. The temperature is being swept over $T \in \{1,2,3,4\}$
and selected by validation accuracy in `results/temperature_sweep.md`.

---

## Quick comparison

| Model | Params | Pre-trained on | Fine-tuned on | Test acc (beans) |
|---|---:|---|---|---:|
| Teacher: `merve/beans-vit-224` (ViT-B/16) | **85.8 M** | ImageNet-21k (self-supervised, 14 M imgs) | beans (3 epochs, lr 5e-5, batch 64) | **0.9375** |
| Student: MobileNetV2 (this repo) | **2.23 M** | — (random init) | beans + distillation from teacher | 0.6797 *(T=5 baseline run)* |

The student has **~2.6 %** of the teacher's parameter count.
