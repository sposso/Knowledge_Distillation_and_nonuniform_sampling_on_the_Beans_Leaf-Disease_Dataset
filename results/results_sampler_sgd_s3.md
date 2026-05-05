# Distillation + Saliency Sampler — beans dataset

Student: ImageNet-pretrained MobileNetV2 wrapped in a Recasens Saliency Sampler (truncated MobileNetV3-Small saliency net), distilled from `merve/beans-vit-224`.

Best checkpoint: `my-awesome-model-sampler-sgd-s3/checkpoint-660` (selected on validation accuracy = 0.9925).

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Temperature (T) | 1.0 |
| Distillation weight (λ) | 0.5 |
| Loss | (1−λ)·CE + λ·T²·KL(student ∥ teacher) |
| Saliency net | MobileNetV3-Small `features[:9]` (~190 K params) |
| Sampler grid | 31×31, padding 30, Gaussian FWHM 13 |
| Blur schedule | always blur first 10 epochs (p=0), no blur after (p=1) |
| Optimizer | AdamW, base lr 5e-5, multipliers task=1.0 / conv_last=0.01 / saliency=0.001 |

## Student metrics (best checkpoint)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| train | 1034 | 0.9961 | 0.9961 |
| validation | 133 | 0.9850 | 0.9850 |
| test | 128 | 0.9766 | 0.9767 |

## Teacher baseline — `merve/beans-vit-224` (inference only)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| test | 128 | 0.9375 | 0.9379 |

Δ test (teacher − student): acc = -0.0391, macro-F1 = -0.0388.

Classes: angular_leaf_spot, bean_rust, healthy
