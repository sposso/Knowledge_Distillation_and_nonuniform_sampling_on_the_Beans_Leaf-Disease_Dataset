# Distillation results — beans dataset

Student: MobileNetV2 (initialized from ImageNet-1k weights `google/mobilenet_v2_1.0_224`) distilled from teacher `merve/beans-vit-224`.
Best checkpoint: `mobilenetv2-imagenet-distill/checkpoint-136` (selected on validation accuracy).

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Temperature (T) | 1.0 |
| Distillation weight (λ) | 0.5 |
| Loss | (1−λ)·CE + λ·T²·KL(student ∥ teacher) |

## Student metrics (best checkpoint)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| train | 1034 | 0.9971 | 0.9971 |
| validation | 133 | 0.9398 | 0.9399 |
| test | 128 | 0.8984 | 0.8989 |

## Teacher baseline — `merve/beans-vit-224` (inference only)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| test | 128 | 0.9375 | 0.9379 |

Distillation gap on test: Δacc = +0.0391, Δmacro-F1 = +0.0390 (teacher − student).

Classes: angular_leaf_spot, bean_rust, healthy

