# Distillation results — beans dataset

Student: MobileNetV2 (from scratch) distilled from teacher `merve/beans-vit-224`.
Best checkpoint: `/home/sposso22/Documents/project/sweep_T_imagenet/T3/checkpoint-442` (selected on validation accuracy).

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Temperature (T) | 3.0 |
| Distillation weight (λ) | 0.5 |
| Loss | (1−λ)·CE + λ·T²·KL(student ∥ teacher) |

## Student metrics (best checkpoint)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| train | 1034 | 0.9981 | 0.9981 |
| validation | 133 | 0.9248 | 0.9247 |
| test | 128 | 0.8906 | 0.8909 |

## Teacher baseline — `merve/beans-vit-224` (inference only)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| test | 128 | 0.9375 | 0.9379 |

Distillation gap on test: Δacc = +0.0469, Δmacro-F1 = +0.0471 (teacher − student).

Classes: angular_leaf_spot, bean_rust, healthy

