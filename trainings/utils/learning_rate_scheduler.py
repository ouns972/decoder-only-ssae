from enum import Enum

from torch.optim import Optimizer


class LRSchedulerTypes(Enum):
    LINEAR = "linear"


class LRScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        lr_scheduler_type: LRSchedulerTypes,
        lr_scheduler_lr_init_linear: float,
        lr_scheduler_lr_final_linear: float,
        n_epochs: int,
    ):
        self.optimizer = optimizer
        self.type = LRSchedulerTypes[lr_scheduler_type]
        self.n_epochs = n_epochs

        # linear scheduler
        self.lr_scheduler_lr_init_linear = lr_scheduler_lr_init_linear
        self.lr_scheduler_lr_final_linear = lr_scheduler_lr_final_linear

        self.current_index = 0

    def step(self, current_iter):
        if self.type == LRSchedulerTypes.LINEAR:
            lr = (
                self.lr_scheduler_lr_init_linear
                + (self.lr_scheduler_lr_final_linear - self.lr_scheduler_lr_init_linear)
                * current_iter
                / self.n_epochs
            )
        self.update_lr(lr)

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
