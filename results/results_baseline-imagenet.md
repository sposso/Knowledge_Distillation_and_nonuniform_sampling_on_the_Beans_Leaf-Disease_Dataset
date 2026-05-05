# No-KD baseline results — beans dataset

Student: MobileNetV2 initialized from ImageNet-1k weights `google/mobilenet_v2_1.0_224`,
fine-tuned on the beans training set with plain cross-entropy (no teacher,
no distillation).
Best checkpoint: `baseline-imagenet/checkpoint-442` (selected on validation accuracy).

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Loss | Cross-entropy only (no distillation) |
| Optimizer | AdamW, lr=5e-5 |
| Per-device batch size | 32 (effective 64 across 2 GPUs) |
| Epochs | 30 |
| Precision | fp16 |
| Seed | 42 |

## Student metrics (best checkpoint)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| train | 1034 | 1.0000 | 1.0000 |
| validation | 133 | 0.9474 | 0.9472 |
| test | 128 | 0.8984 | 0.8992 |

## Teacher reference — `merve/beans-vit-224` (inference only)

| Split | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| test | 128 | 0.9375 | 0.9379 |

Gap on test: Δacc = +0.0391, Δmacro-F1 = +0.0387 (teacher − student).

Classes: angular_leaf_spot, bean_rust, healthy
