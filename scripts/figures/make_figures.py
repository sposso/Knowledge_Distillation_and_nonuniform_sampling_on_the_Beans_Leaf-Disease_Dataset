"""Generate the three Figures/ plots used by the paper:
  1) class distribution in the training set
  2) one random example per class
  3) training vs validation loss across epochs

The loss-curves plot reads the trainer_state.json from a finished run.

Usage (from project root):
    python scripts/figures/make_figures.py \\
        --trainer_state my-awesome-model/checkpoint-510/trainer_state.json
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset


FIG_DIR = Path("Figures")


def class_distribution(train_ds, label_names):
    counts = Counter(train_ds["labels"])
    ordered = [counts[i] for i in range(len(label_names))]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(label_names, ordered, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_title("Beans training set - class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    for bar, n in zip(bars, ordered):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(n), ha="center", va="bottom")
    ax.set_ylim(0, max(ordered) * 1.12)
    fig.tight_layout()
    out = FIG_DIR / "class_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  counts={dict(zip(label_names, ordered))}")


def random_example_per_class(train_ds, label_names, seed: int = 0):
    rng = random.Random(seed)
    by_class = {i: [] for i in range(len(label_names))}
    for idx, lbl in enumerate(train_ds["labels"]):
        by_class[lbl].append(idx)

    fig, axes = plt.subplots(1, len(label_names), figsize=(4 * len(label_names), 4))
    for ax, cls_idx in zip(axes, range(len(label_names))):
        idx = rng.choice(by_class[cls_idx])
        ax.imshow(train_ds[idx]["image"])
        ax.set_title(f"{label_names[cls_idx]}\n(train idx {idx})")
        ax.axis("off")
    fig.suptitle("Random example per class")
    fig.tight_layout()
    out = FIG_DIR / "samples_per_class.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def loss_curves(trainer_state_path: Path):
    state = json.loads(trainer_state_path.read_text())
    history = state["log_history"]

    # Trainer interleaves train-loss and eval-loss records; "loss" appears
    # in both kinds, but eval records also carry "eval_loss".
    train = [(h["epoch"], h["loss"]) for h in history if "loss" in h and "eval_loss" not in h]
    val = [(h["epoch"], h["eval_loss"]) for h in history if "eval_loss" in h]

    fig, ax = plt.subplots(figsize=(8, 5))
    if train:
        xs, ys = zip(*train)
        ax.plot(xs, ys, marker="o", label="train loss", color="#4C72B0")
    if val:
        xs, ys = zip(*val)
        ax.plot(xs, ys, marker="s", label="validation loss", color="#C44E52")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Distillation - training vs. validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "loss_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  (train epochs: {len(train)}, val epochs: {len(val)})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trainer_state", type=str, required=True,
                   help="Path to a trainer_state.json from a finished run "
                        "(e.g. my-awesome-model/checkpoint-510/trainer_state.json).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the random per-class example picks.")
    return p.parse_args()


def main():
    args = parse_args()
    FIG_DIR.mkdir(exist_ok=True)

    dataset = load_dataset("beans")
    train_ds = dataset["train"]
    label_names = train_ds.features["labels"].names

    class_distribution(train_ds, label_names)
    random_example_per_class(train_ds, label_names, seed=args.seed)
    loss_curves(Path(args.trainer_state))


if __name__ == "__main__":
    main()
