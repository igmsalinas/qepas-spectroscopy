"""Learning-rate schedules and optimizer factories."""

from __future__ import annotations

import tensorflow as tf
from keras import optimizers


def make_lr_schedule(
    schedule_name: str,
    initial_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-6,
    decay_rate: float = 0.96,
    decay_steps: int | None = None,
    *,
    steps_per_epoch: int = 1,
):
    """Build a Keras learning-rate schedule.

    Keras schedules are called once per optimizer step, so epoch-based settings
    are converted with ``steps_per_epoch``. Keeping that conversion here avoids
    accidentally completing a 100-epoch decay in the first few batches.
    """
    if initial_lr <= 0:
        raise ValueError("initial_lr must be positive")
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if warmup_epochs < 0 or warmup_epochs >= total_epochs:
        raise ValueError("warmup_epochs must be in [0, total_epochs)")
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")

    warmup_steps = warmup_epochs * steps_per_epoch
    scheduled_steps = max(1, (total_epochs - warmup_epochs) * steps_per_epoch)

    if schedule_name == "constant":
        base: float | optimizers.schedules.LearningRateSchedule = initial_lr
    elif schedule_name == "exponential_decay":
        base = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps or steps_per_epoch,
            decay_rate=decay_rate,
            staircase=False,
        )
    elif schedule_name == "cosine_decay":
        base = optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=scheduled_steps,
            alpha=min_lr / initial_lr,
        )
    elif schedule_name == "cosine_decay_restarts":
        base = optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=max(1, scheduled_steps // 3),
            t_mul=2.0,
            m_mul=0.9,
            alpha=min_lr / initial_lr,
        )
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    if warmup_steps == 0:
        return base

    class WarmupSchedule(optimizers.schedules.LearningRateSchedule):
        def __init__(self) -> None:
            super().__init__()
            self.base_schedule = base
            self.warmup_steps = warmup_steps
            self.initial_lr = initial_lr
            self.is_constant = not isinstance(
                base,
                optimizers.schedules.LearningRateSchedule,
            )

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            boundary = tf.cast(self.warmup_steps, tf.float32)
            warmup_lr = self.initial_lr * step / boundary
            if self.is_constant:
                return tf.cond(
                    step < boundary,
                    lambda: warmup_lr,
                    lambda: tf.constant(self.initial_lr, dtype=tf.float32),
                )
            return tf.cond(
                step < boundary,
                lambda: warmup_lr,
                lambda: self.base_schedule(step - boundary),
            )

        def get_config(self) -> dict[str, object]:
            return {
                "base_schedule": self.base_schedule,
                "warmup_steps": self.warmup_steps,
                "initial_lr": self.initial_lr,
            }

    return WarmupSchedule()


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
    if optimizer_name == "adamw":
        return optimizers.AdamW(weight_decay=weight_decay, **kwargs)
    if optimizer_name == "sgd":
        return optimizers.SGD(momentum=0.9, **kwargs)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")
