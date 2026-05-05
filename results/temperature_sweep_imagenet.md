# Temperature sweep — distillation (imagenet)

Student init: `google/mobilenet_v2_1.0_224`.  
Per-device train batch: 32 (effective 64 on 2 GPUs).  
Epochs: 30.  λ = 0.5.  Selection metric: validation accuracy.

| Temperature | Best val accuracy | Best epoch | Best checkpoint |
|---:|---:|---:|---|
| 1 | 0.9549 | 8.0 | `/home/sposso22/Documents/project/sweep_T_imagenet/T1/checkpoint-136` |
| 2 | 0.9774 | 14.0 | `/home/sposso22/Documents/project/sweep_T_imagenet/T2/checkpoint-238` |
| 3 | 0.9699 | 26.0 | `/home/sposso22/Documents/project/sweep_T_imagenet/T3/checkpoint-442` |
| 4 | 0.9699 | 26.0 | `/home/sposso22/Documents/project/sweep_T_imagenet/T4/checkpoint-442` |
| 5 | 0.9850 | 14.0 | `/home/sposso22/Documents/project/sweep_T_imagenet/T5/checkpoint-238` |

**Best temperature by validation accuracy: T = 5** (val_acc = 0.9850).

