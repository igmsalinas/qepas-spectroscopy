"""Learning-rate schedules with optional warmup."""

from __future__ import annotations

import math
from typing import Callable

import tensorflow as tf
from keras import optimizers


def make_lr_schedule(
    schedule_name: str,
    initial_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-6,
    decay_rate: float = 0.96,
    decay_steps: int = 1,
):
    """Build a Keras learning-rate schedule or a constant float.

    Supported schedules:
      - constant
      - exponential_decay
      - cosine_decay
      - cosine_decay_restarts
    """
    if schedule_name == "constant":
        base = initial_lr
    elif schedule_name == "exponential_decay":
        base = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False,
        )
    elif schedule_name == "cosine_decay":
        base = optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=max(1, total_epochs - warmup_epochs),
            alpha=min_lr / initial_lr if initial_lr > 0 else 0.0,
        )
    elif schedule_name == "cosine_decay_restarts":
        base = optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=max(1, (total_epochs - warmup_epochs) // 3),
            t_mul=2.0,
            m_mul=0.9,
            alpha=min_lr / initial_lr if initial_lr > 0 else 0.0,
        )
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    if warmup_epochs <= 0:
        return base

    class WarmupSchedule(optimizers.schedules.LearningRateSchedule):
        def __init__(self, base_schedule, warmup_steps, initial_lr):
            super().__init__()
            self.base_schedule = base_schedule
            self.warmup_steps = warmup_steps
            self.initial_lr = initial_lr
            self.is_constant = not isinstance(base_schedule, optimizers.schedules.LearningRateSchedule)

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)
            warmup_lr = self.initial_lr * step / warmup_steps
            if self.is_constant:
                return tf.cond(
                    step < warmup_steps,
                    lambda: warmup_lr,
                    lambda: tf.constant(self.initial_lr, dtype=tf.float32),
                )
            return tf.cond(
                step < warmup_steps,
                lambda: warmup_lr,
                lambda: self.base_schedule(step - warmup_steps),
            )

        def get_config(self):
            return {
                "base_schedule": self.base_schedule,
                "warmup_steps": self.warmup_steps,
                "initial_lr": self.initial_lr,
                "is_constant": self.is_constant,
            }

    return WarmupSchedule(base, warmup_epochs, initial_lr)


def make_optimizer(
    optimizer_name: str,
    learning_rate: float | optimizers.schedules.LearningRateSchedule,
    weight_decay: float = 0.0,
    clipnorm: float | None = None,
) -> optimizers.Optimizer:
    """Build Adam, AdamW, or SGD optimizer."""
    kwargs = {"learning_rate": learning_rate}
    if clipnorm is not None:
        kwargs["clipnorm"] = clipnorm

    if optimizer_name == "adam":
        return optimizers.Adam(**kwargs)
    elif optimizer_name == "adamw":
        return optimizers.AdamW(weight_decay=weight_decay, **kwargs)
    elif optimizer_name == "sgd":
        return optimizers.SGD(momentum=0.9, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
