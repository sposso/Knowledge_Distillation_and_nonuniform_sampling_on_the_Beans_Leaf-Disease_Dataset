"""Evaluate a distilled (or baseline) student checkpoint on train/val/test
and write a markdown + JSON report. The teacher is also evaluated on test
as an upper-bound reference.

Usage (from project root):
    python scripts/evaluation/compute_results.py \\
        --run_dir my-awesome-model --temperature 1 --out results/results.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    MobileNetV2ForImageClassification,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default="my-awesome-model",
                   help="Trainer output directory containing checkpoint-*/ folders.")
    p.add_argument("--out", type=str, default=None,
                   help="Markdown output path. Default: results/results_<run>.md")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Distillation T used at training time (for the report).")
    p.add_argument("--lambda_param", type=float, default=0.5,
                   help="Distillation mixing weight used at training time (for the report).")
    return p.parse_args()


def find_best_ckpt(run_dir: Path) -> tuple[str, float]:
    # The Trainer writes the canonical best_model_checkpoint into the
    # last checkpoint's trainer_state.json; we just read it back.
    ckpts = sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not ckpts:
        raise SystemExit(f"No checkpoint-* under {run_dir}")
    state = json.loads((ckpts[-1] / "trainer_state.json").read_text())
    return state["best_model_checkpoint"], float(state["best_metric"])


@torch.no_grad()
def predict(model, dataset, device, batch_size: int = 64):
    loader = DataLoader(
        dataset.with_format("torch", columns=["pixel_values", "labels"]),
        batch_size=batch_size,
        collate_fn=DefaultDataCollator(),
    )
    preds, labels = [], []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
        out = model(pixel_values=pixel_values)
        preds.append(out.logits.argmax(dim=-1).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    best_ckpt, best_val = find_best_ckpt(run_dir)
    print(f"Run dir: {run_dir}")
    print(f"Best checkpoint: {best_ckpt} (val_accuracy={best_val:.4f})")

    processor = AutoImageProcessor.from_pretrained("merve/beans-vit-224")

    def process(examples):
        return processor(examples["image"])

    raw = load_dataset("beans")
    processed = raw.map(process, batched=True)
    label_names = processed["train"].features["labels"].names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2ForImageClassification.from_pretrained(best_ckpt).to(device).half().eval()

    results = {}
    for split in ["train", "validation", "test"]:
        preds, labels = predict(model, processed[split], device)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        results[split] = {"accuracy": acc, "f1_macro": f1, "n": len(labels)}
        print(f"student {split:>10}: n={len(labels):>4}  acc={acc:.4f}  macro-F1={f1:.4f}")

    print("\nLoading teacher (merve/beans-vit-224) for test baseline...")
    teacher = (
        AutoModelForImageClassification.from_pretrained("merve/beans-vit-224")
        .to(device).half().eval()
    )
    t_preds, t_labels = predict(teacher, processed["test"], device)
    teacher_test = {
        "accuracy": accuracy_score(t_labels, t_preds),
        "f1_macro": f1_score(t_labels, t_preds, average="macro"),
        "n": len(t_labels),
    }
    print(
        f"teacher       test: n={teacher_test['n']:>4}  "
        f"acc={teacher_test['accuracy']:.4f}  macro-F1={teacher_test['f1_macro']:.4f}"
    )
    del teacher
    torch.cuda.empty_cache()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    md_path = Path(args.out) if args.out else out_dir / f"results_{run_dir.name}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    md = [
        "# Distillation results - beans dataset",
        "",
        "Student: MobileNetV2 distilled from teacher `merve/beans-vit-224`.",
        f"Best checkpoint: `{best_ckpt}` (selected on validation accuracy).",
        "",
        "## Hyperparameters",
        "",
        "| Hyperparameter | Value |",
        "|---|---|",
        f"| Temperature (T) | {args.temperature} |",
        f"| Distillation weight (lambda) | {args.lambda_param} |",
        "| Loss | (1-lambda)*CE + lambda*T^2*KL(student || teacher) |",
        "",
        "## Student metrics (best checkpoint)",
        "",
        "| Split | N | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for split in ["train", "validation", "test"]:
        r = results[split]
        md.append(f"| {split} | {r['n']} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} |")
    md += [
        "",
        "## Teacher baseline - `merve/beans-vit-224` (inference only)",
        "",
        "| Split | N | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|",
        f"| test | {teacher_test['n']} | {teacher_test['accuracy']:.4f} | "
        f"{teacher_test['f1_macro']:.4f} |",
        "",
    ]
    gap_acc = teacher_test["accuracy"] - results["test"]["accuracy"]
    gap_f1 = teacher_test["f1_macro"] - results["test"]["f1_macro"]
    md.append(
        f"Distillation gap on test: d_acc = {gap_acc:+.4f}, "
        f"d_macro-F1 = {gap_f1:+.4f} (teacher - student)."
    )
    md.append("")
    md.append(f"Classes: {', '.join(label_names)}")
    md_path.write_text("\n".join(md) + "\n")
    print(f"\nWrote {md_path}")

    json_path = md_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "run_dir": str(run_dir),
        "best_checkpoint": best_ckpt,
        "best_val_accuracy_during_training": best_val,
        "temperature": args.temperature,
        "lambda_param": args.lambda_param,
        "label_names": label_names,
        "student_metrics": results,
        "teacher_test_baseline": teacher_test,
    }, indent=2))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
