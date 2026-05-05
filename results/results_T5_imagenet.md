# Distillation results — beans dataset

Student: MobileNetV2 (from scratch) distilled from teacher `merve/beans-vit-224`.
Best checkpoint: `/home/sposso22/Documents/project/sweep_T_imagenet/T5/checkpoint-238` (selected on validation accuracy).

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Temperature (T) | 5.0 |
| Distillation weight (λ) | 0.5 |
| Loss | (1−λ)·CE + λ·T²·KL(student ∥ teacher) |

## Student metrics (best checkpoint)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| train | 1034 | 0.9942 | 0.9942 |
| validation | 133 | 0.9173 | 0.9158 |
| test | 128 | 0.8672 | 0.8674 |

## Teacher baseline — `merve/beans-vit-224` (inference only)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| test | 128 | 0.9375 | 0.9379 |

Distillation gap on test: Δacc = +0.0703, Δmacro-F1 = +0.0705 (teacher − student).

Classes: angular_leaf_spot, bean_rust, healthy

