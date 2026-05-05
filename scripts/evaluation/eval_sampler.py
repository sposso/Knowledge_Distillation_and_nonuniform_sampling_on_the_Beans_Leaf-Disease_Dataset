"""Evaluate a sampler-distilled checkpoint on train/val/test plus a teacher
baseline. Like compute_results.py, but rebuilds the StudentWithSampler
wrapper before loading the checkpoint state_dict (the Trainer can't
auto-load a custom nn.Module from a HF checkpoint).

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \\
        python scripts/evaluation/eval_sampler.py \\
        --run_dir my-awesome-model-sampler-sgd-s3 \\
        --temperature 1 --saliency_scale 3 \\
        --out results/results_sampler_sgd_s3.md
"""

from __future__ import annotations

import argparse
import json
import sys
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
from safetensors.torch import load_file as safetensors_load_file

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.training.main_with_sampler import StudentWithSampler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lambda_param", type=float, default=0.5)
    p.add_argument("--task_input_size", type=int, default=224)
    p.add_argument("--saliency_input_size", type=int, default=224)
    p.add_argument("--saliency_scale", type=float, default=1.0,
                   help="Must match the value used at training time.")
    return p.parse_args()


def find_best_ckpt(run_dir: Path) -> tuple[Path, float]:
    ckpts = sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not ckpts:
        raise SystemExit(f"No checkpoint-* under {run_dir}")
    state = json.loads((ckpts[-1] / "trainer_state.json").read_text())
    return Path(state["best_model_checkpoint"]), float(state.get("best_metric") or 0.0)


@torch.no_grad()
def predict_sampler(model: StudentWithSampler, dataset, device, batch_size: int = 64):
    loader = DataLoader(
        dataset.with_format("torch", columns=["pixel_values_hires", "labels"]),
        batch_size=batch_size,
        collate_fn=DefaultDataCollator(),
    )
    preds, labels = [], []
    for batch in loader:
        pv = batch["pixel_values_hires"].to(device, dtype=torch.float16)
        out = model(pixel_values_hires=pv, p=1.0)  # p=1: never blur at eval
        preds.append(out.logits.argmax(dim=-1).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels)


@torch.no_grad()
def predict_uniform(model, dataset, device, batch_size: int = 64):
    loader = DataLoader(
        dataset.with_format("torch", columns=["pixel_values", "labels"]),
        batch_size=batch_size,
        collate_fn=DefaultDataCollator(),
    )
    preds, labels = [], []
    for batch in loader:
        pv = batch["pixel_values"].to(device, dtype=torch.float16)
        out = model(pixel_values=pv)
        preds.append(out.logits.argmax(dim=-1).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    best_ckpt, best_val = find_best_ckpt(run_dir)
    print(f"Best checkpoint: {best_ckpt} (val_accuracy={best_val:.4f})")

    processor = AutoImageProcessor.from_pretrained("merve/beans-vit-224")
    image_mean = np.array(processor.image_mean, dtype=np.float32).reshape(3, 1, 1)
    image_std = np.array(processor.image_std, dtype=np.float32).reshape(3, 1, 1)

    def process(examples):
        out = processor(examples["image"])
        hires = []
        for pil_img in examples["image"]:
            arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)
            arr = (arr - image_mean) / image_std
            hires.append(arr.astype(np.float32))
        out["pixel_values_hires"] = hires
        return out

    raw = load_dataset("beans")
    processed = raw.map(process, batched=True)
    label_names = processed["train"].features["labels"].names
    num_labels = len(label_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct the wrapper as in training, then overlay the saved weights.
    student_hf = MobileNetV2ForImageClassification.from_pretrained(
        "google/mobilenet_v2_1.0_224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model = StudentWithSampler(
        student=student_hf,
        task_input_size=args.task_input_size,
        saliency_input_size=args.saliency_input_size,
        saliency_scale=args.saliency_scale,
    )

    sd_path = best_ckpt / "model.safetensors"
    if sd_path.exists():
        state_dict = safetensors_load_file(str(sd_path))
    else:
        state_dict = torch.load(best_ckpt / "pytorch_model.bin", map_location="cpu")
    # strict=False: HF checkpoints don't include task_fn (a closure, not a parameter).
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")

    model = model.to(device).half().eval()

    results = {}
    for split in ["train", "validation", "test"]:
        preds, labels = predict_sampler(model, processed[split], device)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        results[split] = {"accuracy": acc, "f1_macro": f1, "n": len(labels)}
        print(f"student {split:>10}: n={len(labels):>4}  acc={acc:.4f}  macro-F1={f1:.4f}")

    print("\nLoading teacher (merve/beans-vit-224) for test baseline...")
    teacher = (
        AutoModelForImageClassification.from_pretrained("merve/beans-vit-224")
        .to(device).half().eval()
    )
    t_preds, t_labels = predict_uniform(teacher, processed["test"], device)
    teacher_test = {
        "accuracy": accuracy_score(t_labels, t_preds),
        "f1_macro": f1_score(t_labels, t_preds, average="macro"),
        "n": len(t_labels),
    }
    print(
        f"teacher       test: n={teacher_test['n']:>4}  "
        f"acc={teacher_test['accuracy']:.4f}  macro-F1={teacher_test['f1_macro']:.4f}"
    )

    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# Distillation + Saliency Sampler - beans dataset",
        "",
        "Student: ImageNet-pretrained MobileNetV2 wrapped in a Recasens "
        "Saliency Sampler (truncated MobileNetV3-Small saliency net), "
        "distilled from `merve/beans-vit-224`.",
        "",
        f"Best checkpoint: `{best_ckpt}` "
        f"(selected on validation accuracy = {best_val:.4f}).",
        "",
        "## Hyperparameters",
        "",
        "| Hyperparameter | Value |",
        "|---|---|",
        f"| Temperature (T) | {args.temperature} |",
        f"| Distillation weight (lambda) | {args.lambda_param} |",
        f"| Saliency scale | {args.saliency_scale} |",
        "| Loss | (1-lambda)*CE + lambda*T^2*KL(student || teacher) |",
        "| Saliency net | MobileNetV3-Small features[:9] (~190 K params) |",
        "| Sampler grid | 31x31, padding 30, Gaussian FWHM 13 |",
        "| Optimizer | SGD, paper multipliers (task=1.0 / conv_last=0.01 / saliency=0.001) |",
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
    md.append(f"d test (teacher - student): acc = {gap_acc:+.4f}, macro-F1 = {gap_f1:+.4f}.")
    md.append("")
    md.append(f"Classes: {', '.join(label_names)}")
    out_md.write_text("\n".join(md) + "\n")
    print(f"\nWrote {out_md}")

    out_json = out_md.with_suffix(".json")
    out_json.write_text(json.dumps({
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_ckpt),
        "best_val_accuracy_during_training": best_val,
        "temperature": args.temperature,
        "lambda_param": args.lambda_param,
        "saliency_scale": args.saliency_scale,
        "label_names": label_names,
        "student_metrics": results,
        "teacher_test_baseline": teacher_test,
    }, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
