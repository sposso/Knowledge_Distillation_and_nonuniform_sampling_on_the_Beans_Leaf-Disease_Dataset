# Distillation vs. no-distillation baseline

Both runs use the **same** student (MobileNetV2 from scratch, ~2.23 M params),
the **same** data (beans), the **same** training config (per-device batch 32,
30 epochs, fp16, DDP across 2 GPUs, seed 42, lr default 5e-5, eval/checkpoint
every epoch, best model loaded by validation accuracy).

The only difference is the loss:

| Run | Loss |
|---|---|
| **Distilled** (T = 1, λ = 0.5) | $(1{-}λ)\,\text{CE} + λ\,T^2\,\text{KL}\big(\sigma(s/T) \,\|\, \sigma(t/T)\big)$ — teacher: `merve/beans-vit-224` |
| **Baseline** | plain cross-entropy |

## Results

| Metric | Baseline (no distill) | Distilled (T = 1) | Δ (distill − baseline) |
|---|---:|---:|---:|
| Best val accuracy (during training) | 0.7895 | 0.7744 | −0.015 |
| Train accuracy (best ckpt, fp16 inference) | 0.9720 | 0.9603 | −0.012 |
| **Test accuracy** | **0.6094** | **0.7188** | **+0.109** |
| **Test macro-F1** | **0.6057** | **0.7202** | **+0.115** |
| Best epoch (by val) | 26 | 26 | — |
| Train wall-clock (s) | 616 | ~706 | +~90 |
| Best checkpoint | `baseline/checkpoint-442` | `sweep_T/T1_b32/checkpoint-442` | — |

Teacher upper bound on test (inference only): **0.9375 acc / 0.9379 F1**.

## Takeaway

- The baseline reaches a **slightly higher peak validation accuracy** (0.7895 vs.
  0.7744), but **generalizes much worse on the held-out test set**: it loses
  about **11 percentage points of test accuracy and macro-F1** relative to the
  distilled student.
- This is the classic distillation effect: the teacher's *soft targets* act as
  a regularizer that produces flatter, more transferable solutions even when
  the student fits training (and one validation snapshot) almost as well
  without them.
- **Distillation helps:** at T = 1, λ = 0.5, the distilled MobileNetV2 closes
  ~52 % of the gap between the no-distill baseline and the teacher
  ((0.7188 − 0.6094) / (0.9375 − 0.6094) ≈ 0.33 of the remaining accuracy gap).
