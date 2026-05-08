"""Distillation with the Recasens Saliency Sampler in front of the MobileNetV2
student.

  raw 500x500  -->  SaliencySampler  -->  warped 224x224  -->  student
                                                                  |
                                                                  v
                                                              student logits
                                          (teacher sees the standard uniform 224x224)


"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    MobileNetV2ForImageClassification,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import ImageClassifierOutput

# Make sibling package `scripts.saliency` importable when launched via torchrun.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.saliency import SaliencySampler, saliency_network_mobilenetv3_small


class StudentWithSampler(nn.Module):
   

    def __init__(
        self,
        student: MobileNetV2ForImageClassification,
        task_input_size: int = 224,
        saliency_input_size: int = 224,
        saliency_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.student = student
        saliency_net = saliency_network_mobilenetv3_small(pretrained=True)
        self.sampler = SaliencySampler(
            task_fn=lambda y: self.student(pixel_values=y).logits,
            saliency_network=saliency_net,
            saliency_channels=48,
            task_input_size=task_input_size,
            saliency_input_size=saliency_input_size,
            saliency_scale=saliency_scale,
        )

    def forward(
        self,
        pixel_values_hires: torch.Tensor,
        labels: torch.Tensor | None = None,
        p: float = 1.0,
        pixel_values: torch.Tensor | None = None,  # accepted but unused (teacher consumes it)
        **_: object,
    ) -> ImageClassifierOutput:
        logits, _x_sampled, _sal = self.sampler(pixel_values_hires, p=p)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return ImageClassifierOutput(loss=loss, logits=logits)


class SamplerDistilTrainer(Trainer):
    """Distillation trainer with paper-exact SGD over three parameter groups
    (task / conv_last / saliency) and the blur schedule from Recasens §3.3."""

    def __init__(
        self,
        teacher_model,
        student_model,
        temperature: float,
        lambda_param: float,
        blur_epochs: int,
        momentum: float = 0.9,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.teacher.to(self.args.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.temperature = temperature
        self.lambda_param = lambda_param
        self.blur_epochs = blur_epochs
        self.momentum = momentum

    def create_optimizer(self):
        # Recasens et al. tune the saliency net much more gently than the
        # task net. Multipliers: task=1.0, conv_last=0.01, saliency=0.001.
        if self.optimizer is None:
            base_lr = self.args.learning_rate
            wd = self.args.weight_decay
            m = self.model.module if hasattr(self.model, "module") else self.model
            param_groups = [
                {"params": list(m.student.parameters()),
                 "lr": base_lr * 1.0,    "name": "task"},
                {"params": list(m.sampler.conv_last.parameters()),
                 "lr": base_lr * 0.01,   "name": "conv_last"},
                {"params": list(m.sampler.localization.parameters()),
                 "lr": base_lr * 0.001,  "name": "saliency"},
            ]
            self.optimizer = torch.optim.SGD(
                param_groups, lr=base_lr, momentum=self.momentum, weight_decay=wd,
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Blur is enabled (p=0) for the first `blur_epochs`, then disabled (p=1).
        epoch = self.state.epoch or 0.0
        p_blur = 0.0 if epoch < self.blur_epochs else 1.0

        pixel_values_hires = inputs["pixel_values_hires"]
        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]

        student_output = model(
            pixel_values_hires=pixel_values_hires,
            labels=labels,
            p=p_blur,
        )

        with torch.no_grad():
            teacher_output = self.teacher(pixel_values=pixel_values)

        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)
        student_target_loss = student_output.loss

        loss = (1.0 - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
        return (loss, student_output) if return_outputs else loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lambda_param", type=float, default=0.5,
                   help="Set to 0 to disable distillation (sampler-only control).")
    # SGD hyperparameters from Recasens et al.
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--output_dir", type=str, default="my-awesome-model-sampler")
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--task_input_size", type=int, default=224)
    p.add_argument("--saliency_input_size", type=int, default=224)
    p.add_argument("--saliency_scale", type=float, default=1.0,
                   help="Multiplier on pre-softmax saliency logits. >1 sharpens the warp.")
    p.add_argument("--blur_epochs", type=int, default=10)
    p.add_argument("--run_name", type=str, default="distillation_sampler")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_test_eval", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    teacher_processor = AutoImageProcessor.from_pretrained("merve/beans-vit-224")
    image_mean = np.array(teacher_processor.image_mean, dtype=np.float32).reshape(3, 1, 1)
    image_std = np.array(teacher_processor.image_std, dtype=np.float32).reshape(3, 1, 1)

    def process(examples):
        # The teacher consumes the standard uniform 224x224 view; the sampler
        # needs the original 500x500 image so it can choose where to zoom.
        teacher_out = teacher_processor(examples["image"])
        hires = []
        for pil_img in examples["image"]:
            arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)
            arr = (arr - image_mean) / image_std
            hires.append(arr.astype(np.float32))
        teacher_out["pixel_values_hires"] = hires
        return teacher_out

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

    student_hf = MobileNetV2ForImageClassification.from_pretrained(
        "google/mobilenet_v2_1.0_224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    student_model = StudentWithSampler(
        student=student_hf,
        task_input_size=args.task_input_size,
        saliency_input_size=args.saliency_input_size,
        saliency_scale=args.saliency_scale,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
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
        # Constant LR matches the paper's effective schedule for the first 30 epochs.
        lr_scheduler_type="constant",
    )

    trainer = SamplerDistilTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=args.temperature,
        lambda_param=args.lambda_param,
        blur_epochs=args.blur_epochs,
        momentum=args.momentum,
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
