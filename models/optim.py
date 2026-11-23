"""
@brief Optimizer / scheduler 
"""

import math
from functools import partial

import torch


def build_optimizer(params, name, lr, weight_decay):
    optimizers = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'RMSprop': torch.optim.RMSprop,
        'SGD': torch.optim.SGD,
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name} (choices: {list(optimizers)})")
    return optimizers[name](params, lr=lr, weight_decay=weight_decay)


def _warmup_decay(epoch, end_warmup, decay_step):
    if epoch <= end_warmup:
        return epoch / end_warmup
    return 0.5 ** ((epoch - end_warmup) // decay_step)


def _warmup_cosine(epoch, end_warmup, total_epochs, min_lr_ratio=0.01):
    if epoch < end_warmup:
        return (epoch + 1) / (end_warmup + 1)
    progress = min(1.0, (epoch - end_warmup) / max(1, total_epochs - end_warmup))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


def build_scheduler(optimizer, name, lr, max_epochs, monitor="train/loss2log"):
    """
    @brief Return the Lightning `lr_scheduler` dict, or None if no scheduler.
            All schedulers are stepped every epoch.
    """
    if name in (None, '', 'None'):
        return None

    d = {"interval": "epoch", "frequency": 1}

    if name == 'StepLR':
        d["scheduler"] = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    elif name == 'ReduceLROnPlateau':
        d["scheduler"] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-7, cooldown=5)
        d["monitor"] = monitor

    elif name == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr / 10, max_lr=lr,
                                                      cycle_momentum=False,
                                                      step_size_up=10, step_size_down=10)
        # workaround: the weakref scale function cannot be pickled in checkpoints (older PyTorch)
        if hasattr(scheduler, "_scale_fn_ref"):
            scheduler._scale_fn_custom = scheduler._scale_fn_ref()
            scheduler._scale_fn_ref = None
        d["scheduler"] = scheduler

    elif name == 'WarmupDecay':
        d["scheduler"] = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(_warmup_decay, end_warmup=10, decay_step=20))

    elif name == 'Warmup':  # linear warmup + cosine decay
        d["scheduler"] = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(_warmup_cosine, end_warmup=7, total_epochs=max_epochs))

    elif name == 'Cosine':  # LinearLR warmup + CosineAnnealing
        warmup_epochs = 10
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                                                   total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup_epochs),
                                                            eta_min=1e-6)
        d["scheduler"] = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine],
                                                               milestones=[warmup_epochs])

    elif name == 'CosineWarmRestart':
        d["scheduler"] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1)

    else:
        raise ValueError(f"Unknown scheduler: {name}")

    return d
