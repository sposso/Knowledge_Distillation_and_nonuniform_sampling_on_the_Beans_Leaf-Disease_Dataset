"""Run a temperature sweep T in {1..5} sequentially on both GPUs and emit
results/temperature_sweep.{md,json} + Figures/temperature_sweep.png.

Run from the project root:
    python scripts/sweeps/sweep_temperature.py                           # random-init sweep
    python scripts/sweeps/sweep_temperature.py \\
        --student_pretrained google/mobilenet_v2_1.0_224 \\
        --sweep_dir sweep_T_imagenet --out_tag imagenet                  # ImageNet-init sweep
    python scripts/sweeps/sweep_temperature.py --summarize-only          # rebuild md/png from existing runs
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT = Path(".").resolve()
RESULTS_DIR = PROJECT / "results"
FIG_DIR = PROJECT / "Figures"


def run_one(temperature: int, sweep_dir: Path, per_device_batch: int,
            num_epochs: int, student_pretrained: str | None,
            run_name_prefix: str) -> Path:
    out = sweep_dir / f"T{temperature}"
    if out.exists():
        # Always start fresh so a partial previous run doesn't pollute the new one.
        shutil.rmtree(out)
    out.mkdir(parents=True)
    log_path = out / "train.log"

    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "4",
        # Required on dual-GPU rigs without NVLink to avoid NCCL P2P errors.
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
    })

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=2", "scripts/training/main.py",
        "--temperature", str(temperature),
        "--output_dir", str(out),
        "--per_device_train_batch_size", str(per_device_batch),
        "--num_train_epochs", str(num_epochs),
        "--run_name", f"{run_name_prefix}_T{temperature}",
        "--skip_test_eval",
    ]
    if student_pretrained:
        cmd += ["--student_pretrained", student_pretrained]
    print(f"\n=== Sweep T={temperature} ===")
    print("cmd:", " ".join(cmd))
    print("log:", log_path)
    with open(log_path, "wb") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        print(f"!! T={temperature} returned {proc.returncode}; check {log_path}")
    return out


def summarize_one(run_dir: Path):
    """Read trainer_state.json from the last checkpoint and extract history."""
    if not run_dir.exists():
        return None
    ckpts = sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not ckpts:
        return None
    state = json.loads((ckpts[-1] / "trainer_state.json").read_text())
    val_history = [(h["epoch"], h["eval_accuracy"], h["eval_loss"])
                   for h in state["log_history"] if "eval_accuracy" in h]
    train_history = [(h["epoch"], h["loss"])
                     for h in state["log_history"] if "loss" in h and "eval_loss" not in h]
    return {
        "best_metric": state.get("best_metric"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
        "val_history": val_history,
        "train_history": train_history,
    }


def write_summary(records, per_device_batch, num_epochs, out_tag, student_pretrained):
    md = []
    title_suffix = f" ({out_tag})" if out_tag else ""
    md.append(f"# Temperature sweep - distillation{title_suffix}")
    md.append("")
    md.append(f"Student init: `{student_pretrained}`.  " if student_pretrained
              else "Student init: random (from scratch).  ")
    md.append(f"Per-device train batch: {per_device_batch} (effective {per_device_batch * 2} on 2 GPUs).  ")
    md.append(f"Epochs: {num_epochs}.  lambda = 0.5.  Selection metric: validation accuracy.")
    md.append("")
    md.append("| Temperature | Best val accuracy | Best epoch | Best checkpoint |")
    md.append("|---:|---:|---:|---|")

    best_T, best_acc = None, -1.0
    for T, rec in records.items():
        if rec is None:
            md.append(f"| {T} | (failed) | - | - |")
            continue
        # Match the recorded best_metric back to its epoch in val_history.
        best_epoch = next(
            (e for e, a, _ in rec["val_history"] if abs(a - rec["best_metric"]) < 1e-9),
            rec["val_history"][-1][0] if rec["val_history"] else None,
        )
        md.append(f"| {T} | {rec['best_metric']:.4f} | {best_epoch} | `{rec['best_model_checkpoint']}` |")
        if rec["best_metric"] > best_acc:
            best_acc = rec["best_metric"]
            best_T = T
    md.append("")
    if best_T is not None:
        md.append(f"**Best temperature by validation accuracy: T = {best_T}** (val_acc = {best_acc:.4f}).")
    md.append("")

    name = "temperature_sweep" + (f"_{out_tag}" if out_tag else "")
    out_md = RESULTS_DIR / f"{name}.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_md}")

    out_json = RESULTS_DIR / f"{name}.json"
    out_json.write_text(json.dumps({
        "per_device_train_batch_size": per_device_batch,
        "num_train_epochs": num_epochs,
        "lambda_param": 0.5,
        "student_pretrained": student_pretrained,
        "best_temperature": best_T,
        "best_val_accuracy": best_acc,
        "runs": {
            str(T): (None if rec is None else {
                "best_metric": rec["best_metric"],
                "best_model_checkpoint": rec["best_model_checkpoint"],
                "val_history": rec["val_history"],
                "train_history": rec["train_history"],
            })
            for T, rec in records.items()
        },
    }, indent=2))
    print(f"Wrote {out_json}")
    return best_T, best_acc


def plot_sweep(records, out_tag):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for (T, rec), c in zip(records.items(), colors):
        if rec is None or not rec["val_history"]:
            continue
        epochs, accs, _ = zip(*rec["val_history"])
        ax1.plot(epochs, accs, marker="o", label=f"T={T}", color=c)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation accuracy")
    ax1.set_title("Validation accuracy by temperature")
    ax1.grid(alpha=0.3)
    ax1.legend()

    Ts = list(records.keys())
    bests = [records[T]["best_metric"] if records[T] else 0 for T in Ts]
    bars = ax2.bar([str(T) for T in Ts], bests, color=colors[:len(Ts)])
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Best validation accuracy")
    ax2.set_title("Best val accuracy per T")
    ax2.set_ylim(0, max(bests) * 1.12 if max(bests) else 1)
    for bar, b in zip(bars, bests):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{b:.3f}", ha="center", va="bottom")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    name = "temperature_sweep" + (f"_{out_tag}" if out_tag else "")
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summarize-only", action="store_true",
                   help="Skip training and just (re)build the summary from existing runs.")
    p.add_argument("--sweep_dir", type=str, default="sweep_T",
                   help="Directory under the project root containing T<temp>/ subruns.")
    p.add_argument("--out_tag", type=str, default="",
                   help="Suffix for output file names (e.g. 'imagenet').")
    p.add_argument("--student_pretrained", type=str, default=None,
                   help="Pass-through to scripts/training/main.py.")
    p.add_argument("--per_device_batch", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--temperatures", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--run_name_prefix", type=str, default="distillation")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    sweep_dir = (PROJECT / args.sweep_dir).resolve()

    if not args.summarize_only:
        sweep_dir.mkdir(exist_ok=True, parents=True)
        for T in args.temperatures:
            run_one(T, sweep_dir, args.per_device_batch, args.num_epochs,
                    args.student_pretrained, args.run_name_prefix)

    records = {T: summarize_one(sweep_dir / f"T{T}") for T in args.temperatures}
    write_summary(records, args.per_device_batch, args.num_epochs,
                  args.out_tag, args.student_pretrained)
    plot_sweep(records, args.out_tag)


if __name__ == "__main__":
    sys.exit(main())
