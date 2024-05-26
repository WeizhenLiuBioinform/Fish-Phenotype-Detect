from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, initial_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linearly increase learning rate
            lr_scale = (self.last_epoch + 1) / self.warmup_epochs
            lr = [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            # After warmup, keep the base learning rate unchanged
            lr = [base_lr for base_lr in self.base_lrs]

        return lr
