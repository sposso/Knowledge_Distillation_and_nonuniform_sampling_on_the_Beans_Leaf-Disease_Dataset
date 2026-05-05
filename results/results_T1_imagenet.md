# Distillation results — beans dataset

Student: MobileNetV2 (from scratch) distilled from teacher `merve/beans-vit-224`.
Best checkpoint: `/home/sposso22/Documents/project/sweep_T_imagenet/T1/checkpoint-136` (selected on validation accuracy).

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Temperature (T) | 1.0 |
| Distillation weight (λ) | 0.5 |
| Loss | (1−λ)·CE + λ·T²·KL(student ∥ teacher) |

## Student metrics (best checkpoint)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| train | 1034 | 0.9990 | 0.9990 |
| validation | 133 | 0.9323 | 0.9326 |
| test | 128 | 0.9219 | 0.9224 |

## Teacher baseline — `merve/beans-vit-224` (inference only)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| test | 128 | 0.9375 | 0.9379 |

Distillation gap on test: Δacc = +0.0156, Δmacro-F1 = +0.0155 (teacher − student).

Classes: angular_leaf_spot, bean_rust, healthy

