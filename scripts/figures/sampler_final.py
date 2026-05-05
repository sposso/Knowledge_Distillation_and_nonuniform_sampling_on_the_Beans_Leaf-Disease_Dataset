"""Render the paper's qualitative sampler figure from explicit test indices.

For each chosen test image (rows), shows four panels:
    input image | saliency map | transformation grid | sampled image

Usage (from project root):
    CUDA_VISIBLE_DEVICES="" python scripts/figures/sampler_final.py \\
        --checkpoint my-awesome-model-sampler-sgd-s3/checkpoint-660 \\
        --indices 21,27,36,52,74 --saliency_scale 3 \\
        --out Figures/sampler_final.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import load_file as safetensors_load_file
from scipy.interpolate import griddata
from transformers import AutoImageProcessor, MobileNetV2ForImageClassification

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.training.main_with_sampler import StudentWithSampler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a checkpoint-N directory inside a sampler run.")
    p.add_argument("--indices", type=str, required=True,
                   help="Comma-separated dataset indices, e.g. 21,27,36,52,74.")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "validation", "test"])
    p.add_argument("--saliency_scale", type=float, default=3.0,
                   help="Must match the value used at training time.")
    p.add_argument(
        "--title", type=str,
        default="Visualization of sampler behavior for the beans fine-grained classification task",
    )
    return p.parse_args()


def saliency_to_uint8(sal: torch.Tensor) -> np.ndarray:
    arr = sal.detach().cpu().float().numpy()[0]
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    return (arr * 255).astype(np.uint8)


def denorm_to_uint8(t: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = t.detach().cpu().float().numpy() * std + mean
    arr = np.clip(arr, 0.0, 1.0)
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


def draw_grid_output_coords(ax, grid_t: torch.Tensor, n_lines: int = 17):
    # Draw the deformation in OUTPUT-image coordinates (Recasens Fig. 3, col 3):
    # cells appear larger where the warp magnifies the input.
    g = grid_t.detach().cpu().float().numpy()
    H, W = g.shape[:2]
    hh, ww = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    points = np.column_stack([g[..., 0].ravel(), g[..., 1].ravel()])
    # A small inset margin avoids extrapolation artifacts where grid_sample
    # has saturated to +/- 1.
    margin = 0.04
    xmin, xmax = float(g[..., 0].min()), float(g[..., 0].max())
    ymin, ymax = float(g[..., 1].min()), float(g[..., 1].max())
    sx = np.linspace(xmin + margin, xmax - margin, n_lines)
    sy = np.linspace(ymin + margin, ymax - margin, n_lines)
    SX, SY = np.meshgrid(sx, sy, indexing="xy")
    targets = np.column_stack([SX.ravel(), SY.ravel()])
    out_w = griddata(points, ww.ravel(), targets, method="cubic").reshape(n_lines, n_lines)
    out_h = griddata(points, hh.ravel(), targets, method="cubic").reshape(n_lines, n_lines)
    for i in range(n_lines):
        m = ~(np.isnan(out_w[i, :]) | np.isnan(out_h[i, :]))
        if m.any():
            ax.plot(out_w[i, m], out_h[i, m], color="black", linewidth=0.6)
    for j in range(n_lines):
        m = ~(np.isnan(out_w[:, j]) | np.isnan(out_h[:, j]))
        if m.any():
            ax.plot(out_w[m, j], out_h[m, j], color="black", linewidth=0.6)
    ax.set_xlim(0, W - 1); ax.set_ylim(H - 1, 0)
    ax.set_aspect("equal"); ax.axis("off")


@torch.no_grad()
def sampler_outputs(sampler, x_hires, scale):
    # Replicate SaliencySampler.forward step-by-step so we can return the
    # intermediate saliency map and grid alongside the sampled image.
    x_low = F.adaptive_avg_pool2d(x_hires, sampler.saliency_input_size)
    feat = F.relu(sampler.localization(x_low))
    sal = sampler.conv_last(feat)
    sal = F.interpolate(sal, size=sampler.grid_size, mode="bilinear", align_corners=True) * scale
    sal_p = F.softmax(sal.flatten(1), dim=1).view_as(sal)
    sal_pad = F.pad(sal_p, [sampler.padding_size] * 4, mode="replicate")
    grid = sampler._create_grid(sal_pad)
    samp = F.grid_sample(x_hires, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sal_p, grid, samp


def main():
    args = parse_args()
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    print(f"Loading {args.checkpoint}")

    proc = AutoImageProcessor.from_pretrained("merve/beans-vit-224")
    mean = np.array(proc.image_mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(proc.image_std, dtype=np.float32).reshape(3, 1, 1)
    ds = load_dataset("beans")[args.split]

    student_hf = MobileNetV2ForImageClassification.from_pretrained(
        "google/mobilenet_v2_1.0_224", num_labels=3, ignore_mismatched_sizes=True
    )
    m = StudentWithSampler(student=student_hf, saliency_scale=args.saliency_scale)
    sd = safetensors_load_file(str(Path(args.checkpoint) / "model.safetensors"))
    m.load_state_dict(sd, strict=False)
    m.eval()
    sampler = m.sampler

    batch = []
    for idx in indices:
        pil = ds[idx]["image"].convert("RGB")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        batch.append(((arr.transpose(2, 0, 1) - mean) / std).astype(np.float32))
    x_hires = torch.from_numpy(np.stack(batch))
    sal, grid, sampled = sampler_outputs(sampler, x_hires, args.saliency_scale)

    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.0 * n))
    if n == 1:
        axes = axes[None, :]
    for c, t in enumerate(["Input image", "Saliency map", "Transformation grid", "Sampled image"]):
        axes[0, c].set_title(t, fontsize=12)

    for r, idx in enumerate(indices):
        row = axes[r]
        pil = ds[idx]["image"].convert("RGB")
        row[0].imshow(np.asarray(pil)); row[0].axis("off")
        row[1].imshow(saliency_to_uint8(sal[r]), cmap="hot", interpolation="bilinear")
        row[1].axis("off")
        draw_grid_output_coords(row[2], grid[r])
        row[3].imshow(denorm_to_uint8(sampled[r], mean, std)); row[3].axis("off")

    fig.suptitle(args.title, fontsize=13, y=1.005)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
