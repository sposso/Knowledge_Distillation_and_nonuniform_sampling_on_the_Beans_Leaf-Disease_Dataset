# Scripts

All scripts assume they are run **from the project root** so that relative
paths like `my-awesome-model/`, `sweep_T/`, `results/`, `Figures/` resolve
correctly.

## Layout

```
scripts/
├── training/               training entry points (run via torchrun)
│   ├── main.py                     distillation, ViT -> MobileNetV2
│   ├── train_baseline.py           MobileNetV2, plain CE (no teacher)
│   └── main_with_sampler.py        distillation + Recasens saliency sampler
│
├── saliency/               SaliencySampler module
│   ├── saliency_sampler.py         the layer (forward + sampling grid)
│   └── saliency_network.py         truncated MobileNetV3-Small backbone
│
├── evaluation/             checkpoint -> markdown + JSON report
│   ├── compute_results.py          for non-sampler runs (uniform 224 input)
│   └── eval_sampler.py             for sampler runs (consumes the 500x500 view)
│
├── sweeps/                 hyperparameter orchestration
│   └── sweep_temperature.py        T in {1..5} sweep + summary md/json/png
│
└── figures/                plot generators
    ├── make_figures.py             class distribution, samples per class, loss curves
    └── sampler_final.py            paper Fig. 1 (sampler qualitative)
```

## Common invocations

```bash
# Distillation (best config: T=1, batch 32, 30 epochs)
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --standalone --nproc_per_node=2 \
    scripts/training/main.py --temperature 1 --output_dir my-awesome-model

# Distillation + saliency sampler (paper-exact SGD, scale = 3)
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --standalone --nproc_per_node=2 \
    scripts/training/main_with_sampler.py \
    --temperature 1 --saliency_scale 3 \
    --output_dir my-awesome-model-sampler-sgd-s3

# No-distillation baseline
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --standalone --nproc_per_node=2 \
    scripts/training/train_baseline.py --output_dir baseline

# Evaluate a non-sampler run on train/val/test
python scripts/evaluation/compute_results.py \
    --run_dir my-awesome-model --temperature 1 --out results/results.md

# Evaluate a sampler run
python scripts/evaluation/eval_sampler.py \
    --run_dir my-awesome-model-sampler-sgd-s3 \
    --saliency_scale 3 --temperature 1 \
    --out results/results_sampler_sgd_s3.md

# Temperature sweep T in {1..5} at batch 32
python scripts/sweeps/sweep_temperature.py \
    --sweep_dir sweep_T --temperatures 1 2 3 4 5 --per_device_batch 32

# Standard data/loss figures
python scripts/figures/make_figures.py \
    --trainer_state my-awesome-model/checkpoint-510/trainer_state.json

# Paper Fig. 1 (sampler qualitative)
CUDA_VISIBLE_DEVICES="" python scripts/figures/sampler_final.py \
    --checkpoint my-awesome-model-sampler-sgd-s3/checkpoint-660 \
    --indices 21,27,36,52,74 --saliency_scale 3 \
    --out Figures/sampler_final.png
```
