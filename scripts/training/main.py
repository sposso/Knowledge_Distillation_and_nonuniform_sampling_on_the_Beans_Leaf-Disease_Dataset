"""Knowledge distillation: ViT teacher (merve/beans-vit-224) -> MobileNetV2
student on the `beans` image-classification dataset.

Run on both local GPUs from the project root:
    torchrun --nproc_per_node=2 scripts/training/main.py [--temperature 5] [--output_dir my-awesome-model]

To initialize the student from ImageNet-1k pretrained weights, add:
    --student_pretrained google/mobilenet_v2_1.0_224
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import load_dataset
import evaluate
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    MobileNetV2Config,
    MobileNetV2ForImageClassification,
    Trainer,
    TrainingArguments,
)


class ImageDistilTrainer(Trainer):
    def __init__(
        self,
        teacher_model,
        student_model,
        temperature: float,
        lambda_param: float,
        *args,
        **kwargs,
    ):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.teacher.to(self.args.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.temperature = temperature
        self.lambda_param = lambda_param

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_output = model(**inputs)
        with torch.no_grad():
            teacher_output = self.teacher(**inputs)

        # Hinton (2015): KL on softened logits, scaled by T^2 so the soft-target
        # gradient stays comparable to the hard-target CE term as T varies.
        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)
        student_target_loss = student_output.loss

        loss = (1.0 - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
        return (loss, student_output) if return_outputs else loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--temperature", type=float, default=5.0)
    p.add_argument("--lambda_param", type=float, default=0.5)
    p.add_argument("--output_dir", type=str, default="my-awesome-model")
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--run_name", type=str, default="distillation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_test_eval", action="store_true")
    p.add_argument(
        "--student_pretrained",
        type=str,
        default=None,
        help="If set (e.g. 'google/mobilenet_v2_1.0_224'), initialize MobileNetV2 "
             "from this checkpoint; otherwise train from scratch.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    teacher_processor = AutoImageProcessor.from_pretrained("merve/beans-vit-224")

    def process(examples):
        return teacher_processor(examples["image"])

    # main_process_first ensures only rank 0 hits the network/cache for
    # download + map; the other ranks reuse the cached arrow file.
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

    teacher_model = AutoModelForImageClassification.from_pretrained(
        "merve/beans-vit-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    if args.student_pretrained:
        student_model = MobileNetV2ForImageClassification.from_pretrained(
            args.student_pretrained,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
    else:
        student_model = MobileNetV2ForImageClassification(MobileNetV2Config(num_labels=num_labels))

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

    trainer = ImageDistilTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=args.temperature,
        lambda_param=args.lambda_param,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed["validation"],
        data_collator=DefaultDataCollator(),
        processing_class=teacher_processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if args.skip_test_eval:
        return

    # Trackio finalizes its run inside train(); strip its callback before the
    # final test eval to avoid logging on a closed run.
    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks
        if "trackio" not in type(cb).__name__.lower()
        and "trackio" not in type(cb).__module__.lower()
    ]
    trainer.args.report_to = []

    test_metrics = trainer.evaluate(processed["test"], metric_key_prefix="test")
    if trainer.is_world_process_zero():
        print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
