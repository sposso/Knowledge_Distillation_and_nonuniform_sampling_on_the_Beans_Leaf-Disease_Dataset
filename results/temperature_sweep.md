# Temperature sweep — distillation

Selection metric: **validation accuracy** (best across 30 epochs).  λ = 0.5.  fp16 + DDP across 2 GPUs.

## Per-device batch = 32 (effective 64)

| Temperature | Best val accuracy | Best epoch | Best checkpoint |
|---:|---:|---:|---|
| 1 | 0.7744 | 26 | `sweep_T/T1_b32/checkpoint-442` |
| 2 | 0.6842 | 14 | `sweep_T/T2_b32/checkpoint-238` |
| 3 | 0.7218 | 26 | `sweep_T/T3_b32/checkpoint-442` |
| 4 | 0.7293 | 14 | `sweep_T/T4_b32/checkpoint-238` |
| 5 | 0.7594 | 14 | `my-awesome-model/checkpoint-238` |

## Per-device batch = 64 (effective 128)

| Temperature | Best val accuracy | Best epoch | Best checkpoint |
|---:|---:|---:|---|
| 1 | 0.7444 | 14 | `/home/sposso22/Documents/project/sweep_T/T1/checkpoint-126` |
| 2 | 0.7068 | 21 | `/home/sposso22/Documents/project/sweep_T/T2/checkpoint-189` |
| 3 | 0.7293 | 14 | `/home/sposso22/Documents/project/sweep_T/T3/checkpoint-126` |
| 4 | 0.7068 | 10 | `/home/sposso22/Documents/project/sweep_T/T4/checkpoint-90` |
| 5 | 0.7444 | 14 | `sweep_T/T5_b64/checkpoint-126` |

## Side-by-side (best val accuracy)

| Temperature | batch 32 | batch 64 | Δ (b32 − b64) |
|---:|---:|---:|---:|
| 1 | 0.7744 | 0.7444 | +0.0301 |
| 2 | 0.6842 | 0.7068 | -0.0226 |
| 3 | 0.7218 | 0.7293 | -0.0075 |
| 4 | 0.7293 | 0.7068 | +0.0226 |
| 5 | 0.7594 | 0.7444 | +0.0150 |

**Overall best:** T = 1, batch = 32 → val_acc = 0.7744 at epoch 26.

