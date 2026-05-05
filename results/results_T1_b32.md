# Distillation results — beans dataset

Student: MobileNetV2 (from scratch) distilled from teacher `merve/beans-vit-224`.
Best checkpoint: `sweep_T/T1_b32/checkpoint-442` (selected on validation accuracy).

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Temperature (T) | 1.0 |
| Distillation weight (λ) | 0.5 |
| Loss | (1−λ)·CE + λ·T²·KL(student ∥ teacher) |

## Student metrics (best checkpoint)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| train | 1034 | 0.9603 | 0.9605 |
| validation | 133 | 0.6842 | 0.6829 |
| test | 128 | 0.7188 | 0.7202 |

## Teacher baseline — `merve/beans-vit-224` (inference only)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| test | 128 | 0.9375 | 0.9379 |

Distillation gap on test: Δacc = +0.2188, Δmacro-F1 = +0.2177 (teacher − student).

Classes: angular_leaf_spot, bean_rust, healthy

