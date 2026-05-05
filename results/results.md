# Distillation results — beans dataset

Student: MobileNetV2 (from scratch) distilled from teacher `merve/beans-vit-224`.
Best checkpoint: `my-awesome-model/checkpoint-238` (selected on validation accuracy).

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Temperature (T) | 5.0 |
| Distillation weight (λ) | 0.5 |
| Loss | (1−λ)·CE + λ·T²·KL(student ∥ teacher) |

## Student metrics (best checkpoint)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| train | 1034 | 0.7679 | 0.7696 |
| validation | 133 | 0.6692 | 0.6640 |
| test | 128 | 0.6797 | 0.6718 |

## Teacher baseline — `merve/beans-vit-224` (inference only)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| test | 128 | 0.9375 | 0.9379 |

Distillation gap on test: Δacc = +0.2578, Δmacro-F1 = +0.2662 (teacher − student).

Classes: angular_leaf_spot, bean_rust, healthy

