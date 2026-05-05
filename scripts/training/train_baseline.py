"""No-distillation baseline: MobileNetV2 trained on `beans` with plain
cross-entropy. Same data and same hyperparameters as the distilled run, so
the only difference vs. main.py is the loss.

Run from the project root, on both GPUs:
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --standalone --nproc_per_node=2 \\
        scripts/training/train_baseline.py [--output_dir baseline] [--per_device_train_batch_size 32]
"""

import argparse

import numpy as np
from accelerate import PartialState
from datasets import load_dataset
import evaluate
from transformers import (
    AutoImageProcessor,
    DefaultDataCollator,
    MobileNetV2Config,
    MobileNetV2ForImageClassification,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="baseline")
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--run_name", type=str, default="baseline_no_distillation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--student_pretrained",
        type=str,
        default=None,
        help="If set (e.g. 'google/mobilenet_v2_1.0_224'), initialize from "
             "this checkpoint; otherwise train from scratch.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Use the teacher's processor so the input pipeline matches main.py exactly.
    processor = AutoImageProcessor.from_pretrained("merve/beans-vit-224")

    def process(examples):
        return processor(examples["image"])

    state = PartialState()
    with state.main_process_first():
        dataset = load_dataset("beans")
        processed = dataset.map(process, batched=True)

    num_labels = len(processed["train"].features["labels"].names)

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        return {
            "accuracy": accuracy_metric.compute(
                references=labels, predictions=np.argmax(predictions, axis=1)
            )["accuracy"]
        }

    if args.student_pretrained:
        student = MobileNetV2ForImageClassification.from_pretrained(
            args.student_pretrained,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
    else:
        student = MobileNetV2ForImageClassification(MobileNetV2Config(num_labels=num_labels))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=True,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="trackio",
        run_name=args.run_name,
        seed=args.seed,
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=student,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed["validation"],
        data_collator=DefaultDataCollator(),
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
