"""Plot how peaked/uncertain the student's softmax outputs are after training
with different distillation temperatures.

For each T's best checkpoint (selected on validation accuracy during the
sweep), we run inference on the chosen split and collect the per-image
softmax over the 3 beans classes. Two distributions per T are then plotted:

  - Max-class probability per image: top-1 confidence at deployment.
  - Softmax entropy per image: how spread the prediction is across classes.

The teacher (always evaluated at T=1) is overlaid as a dashed reference,
so you can see whether higher training temperatures pull the student
distribution towards (or away from) the teacher's calibration.

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \\
        python scripts/figures/logits_distribution.py \\
        --sweep_dir sweep_T_imagenet --split test \\
        --out Figures/logits_distribution.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    MobileNetV2ForImageClassification,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", type=str, default="sweep_T_imagenet",
                   help="Directory containing T<N>/ subdirs (default layout). "
                        "Ignored if --run_dirs is given.")
    p.add_argument("--run_dirs", type=str, nargs="+", default=None,
                   help="Explicit list of run directories, one per --temperatures entry. "
                        "Use this when the sweep dirs don't follow the T<N>/ naming "
                        "(e.g. random-init b32 sweep: sweep_T/T1_b32 ... my-awesome-model).")
    p.add_argument("--temperatures", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "validation", "test"])
    p.add_argument("--out", type=str, default="Figures/logits_distribution.png")
    p.add_argument("--teacher_id", type=str, default="merve/beans-vit-224")
    return p.parse_args()


def find_best_ckpt(run_dir: Path) -> str:
    ckpts = sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not ckpts:
        raise SystemExit(f"No checkpoint-* under {run_dir}")
    state = json.loads((ckpts[-1] / "trainer_state.json").read_text())
    return state["best_model_checkpoint"]


@torch.no_grad()
def collect_softmax(model, dataset, device, batch_size: int = 64) -> np.ndarray:
    """Run the model on the dataset and return softmax probabilities at T=1."""
    loader = DataLoader(
        dataset.with_format("torch", columns=["pixel_values", "labels"]),
        batch_size=batch_size,
        collate_fn=DefaultDataCollator(),
    )
    probs = []
    for batch in loader:
        pv = batch["pixel_values"].to(device, dtype=torch.float16)
        logits = model(pixel_values=pv).logits
        # Always softmax at T=1 here — we want to compare what the trained
        # students actually output at deployment, not the (re)softened signal.
        probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)  # (N, num_classes)


def entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -(p * np.log(p + eps)).sum(axis=-1)


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(args.teacher_id)

    def process(examples):
        return processor(examples["image"])

    raw = load_dataset("beans")
    processed = raw[args.split].map(process, batched=True)
    n_examples = len(processed)
    print(f"Split: {args.split}  N={n_examples}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- Students (one best ckpt per T) -------
    if args.run_dirs is not None:
        if len(args.run_dirs) != len(args.temperatures):
            raise SystemExit("--run_dirs must have the same length as --temperatures")
        run_dirs = [Path(d) for d in args.run_dirs]
    else:
        run_dirs = [sweep_dir / f"T{T}" for T in args.temperatures]

    student_probs = {}
    for T, run_dir in zip(args.temperatures, run_dirs):
        best = find_best_ckpt(run_dir)
        print(f"T={T}: {best}")
        model = MobileNetV2ForImageClassification.from_pretrained(best).to(device).half().eval()
        student_probs[T] = collect_softmax(model, processed, device)
        del model
        torch.cuda.empty_cache()

    # ------- Teacher (one inference, used as reference) -------
    print(f"teacher: {args.teacher_id}")
    teacher = AutoModelForImageClassification.from_pretrained(args.teacher_id).to(device).half().eval()
    teacher_probs = collect_softmax(teacher, processed, device)
    del teacher
    torch.cuda.empty_cache()

    # ------- Plot -------
    fig, (ax_max, ax_ent) = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(args.temperatures) - 1, 1)) for i in range(len(args.temperatures))]

    grid_max = np.linspace(1.0 / 3.0, 1.0, 400)        # 3-class softmax min: 1/3
    grid_ent = np.linspace(0.0, math.log(3.0), 400)    # 3-class entropy max: log(3)

    def kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        # gaussian_kde dies if all values are equal — guard with a tiny jitter.
        if np.allclose(values, values[0]):
            values = values + np.random.normal(0, 1e-6, size=values.shape)
        return gaussian_kde(values)(grid)

    for (T, c) in zip(args.temperatures, colors):
        p = student_probs[T]
        ax_max.plot(grid_max, kde_curve(p.max(axis=-1), grid_max),
                    color=c, linewidth=2.0, label=f"T={T}")
        ax_max.fill_between(grid_max, kde_curve(p.max(axis=-1), grid_max),
                            color=c, alpha=0.08)
        ax_ent.plot(grid_ent, kde_curve(entropy(p), grid_ent),
                    color=c, linewidth=2.0, label=f"T={T}")
        ax_ent.fill_between(grid_ent, kde_curve(entropy(p), grid_ent),
                            color=c, alpha=0.08)

    ax_max.plot(grid_max, kde_curve(teacher_probs.max(axis=-1), grid_max),
                color="black", linewidth=1.6, linestyle="--", label="teacher")
    ax_ent.plot(grid_ent, kde_curve(entropy(teacher_probs), grid_ent),
                color="black", linewidth=1.6, linestyle="--", label="teacher")

    ax_max.set_xlabel("Max-class softmax probability")
    ax_max.set_ylabel("Density")
    ax_max.set_title("Top-1 confidence per image")
    ax_max.set_xlim(1.0 / 3.0, 1.02)
    ax_max.grid(alpha=0.3)
    ax_max.legend(title="Trained with", loc="upper left")

    ax_ent.set_xlabel("Softmax entropy (nats)")
    ax_ent.set_ylabel("Density")
    ax_ent.set_title("Output entropy per image")
    ax_ent.axvline(math.log(3.0), color="grey", linestyle=":", linewidth=1)
    ax_ent.text(math.log(3.0) - 0.02, ax_ent.get_ylim()[1] * 0.95,
                "uniform", color="grey", ha="right", va="top", fontsize=9)
    ax_ent.set_xlim(-0.02, math.log(3.0) + 0.05)
    ax_ent.grid(alpha=0.3)
    ax_ent.legend(title="Trained with", loc="upper left")

    fig.suptitle(
        f"Softmax-output behaviour of the per-T best students  ({args.split} split, N={n_examples})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    # ------- Side report: per-T summary stats -------
    print("\nPer-T stats:")
    print(f"  {'T':>3}  {'mean max-p':>11}  {'mean entropy':>13}  {'top-1 acc':>10}")
    labels = np.asarray(processed["labels"])
    for T in args.temperatures:
        p = student_probs[T]
        acc = (p.argmax(axis=-1) == labels).mean()
        print(f"  {T:>3}  {p.max(axis=-1).mean():>11.4f}  {entropy(p).mean():>13.4f}  {acc:>10.4f}")
    p = teacher_probs
    acc = (p.argmax(axis=-1) == labels).mean()
    print(f"  tch  {p.max(axis=-1).mean():>11.4f}  {entropy(p).mean():>13.4f}  {acc:>10.4f}")


if __name__ == "__main__":
    main()
