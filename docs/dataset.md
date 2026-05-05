# Dataset: `AI-Lab-Makerere/beans`

[Hugging Face Hub link](https://huggingface.co/datasets/AI-Lab-Makerere/beans) ·
[Original GitHub](https://github.com/AI-Lab-Makerere/ibean) ·
License: **MIT**

## Origin

- **Collector:** Makerere AI Lab — Makerere University, **Kampala, Uganda**.
- **Released:** January 2020.
- **Domain:** Smallholder-agriculture computer vision; leaf-disease detection in
  field-grown common beans (*Phaseolus vulgaris*).
- **Annotation:** *expert-generated* labels (per Hub metadata
  `annotations_creators: expert-generated`); language: English.

## Task

Multi-class **image classification** — given a single leaf image, predict
whether it is healthy or which of two fungal diseases is present.

## Classes (3 total)

| Label id | Class | Description |
|---:|---|---|
| 0 | `angular_leaf_spot` | Fungal disease (*Pseudocercospora griseola*) — angular brown lesions bounded by leaf veins. |
| 1 | `bean_rust` | Fungal disease (*Uromyces appendiculatus*) — small reddish-brown pustules on the leaf surface. |
| 2 | `healthy` | No disease. |

## Image specs

| Property | Value |
|---|---|
| Format | JPEG (RGB) |
| Resolution | 500 × 500 px (then resized to 224 × 224 by the ViT processor) |
| Modality | RGB image + integer label |
| Total size on disk | ~360 MB |
| Fields | `image_file_path: str`, `image: PIL.Image`, `labels: int` |

## Splits and counts

| Split | # images | Share |
|---|---:|---:|
| **train** | 1,034 | 79.8 % |
| **validation** | 133 | 10.3 % |
| **test** | 128 | 9.9 % |
| **Total** | **1,295** | 100 % |

### Per-class distribution (training set)

| Class | Count |
|---|---:|
| `angular_leaf_spot` | 345 |
| `bean_rust` | 348 |
| `healthy` | 341 |

The training set is **nearly balanced** across the three classes (≈ 33 % each).
A bar chart of these counts is in `Figures/class_distribution.png`, and one
random training example per class is in `Figures/samples_per_class.png`.

## How it is used in this repo

- Loaded via `datasets.load_dataset("beans")` (alias resolves to
  `AI-Lab-Makerere/beans`).
- Pre-processed with `AutoImageProcessor.from_pretrained("merve/beans-vit-224")`
  → resize, center-crop, and normalize to ViT-Base inputs (224 × 224, mean/std
  from ImageNet).
- The **training** split drives gradient updates; the **validation** split is
  the only signal used for early-stopping / checkpoint selection
  (`metric_for_best_model="accuracy"`); the **test** split is held out and
  reported once at the end.

## Citation

```bibtex
@ONLINE {beansdata,
    author = "Makerere AI Lab",
    title  = "Bean disease dataset",
    month  = "January",
    year   = "2020",
    url    = "https://github.com/AI-Lab-Makerere/ibean/"
}
```
