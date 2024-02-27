import random, torch, os, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import torch

from timm.scheduler.scheduler import Scheduler




def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        """ Cosine Scheduler
        Cosine schedule with warmup.
        Copyright 2021 Ross Wightman
        """
        import math
        import torch

        from timm.scheduler.scheduler import Scheduler

        class CosineScheduler(Scheduler):
            """
            Cosine decay with warmup.
            This is described in the paper https://arxiv.org/abs/1608.03983.

            Modified from timm's implementation.
            """

            def __init__(self,
                         optimizer: torch.optim.Optimizer,
                         param_name: str,
                         t_max: int,
                         value_min: float = 0.,
                         warmup_t=0,
                         const_t=0,
                         initialize=True) -> None:
                super().__init__(
                    optimizer, param_group_field=param_name, initialize=initialize)

                assert t_max > 0
                assert value_min >= 0
                assert warmup_t >= 0
                assert const_t >= 0

                self.cosine_t = t_max - warmup_t - const_t
                self.value_min = value_min
                self.warmup_t = warmup_t
                self.const_t = const_t

                if self.warmup_t:
                    self.warmup_steps = [(v - value_min) / self.warmup_t for v in self.base_values]
                    super().update_groups(self.value_min)
                else:
                    self.warmup_steps = []

            def _get_value(self, t):
                if t < self.warmup_t:
                    values = [self.value_min + t * s for s in self.warmup_steps]
                elif t < self.warmup_t + self.const_t:
                    values = self.base_values
                else:
                    t = t - self.warmup_t - self.const_t

                    value_max_values = [v for v in self.base_values]

                    values = [
                        self.value_min + 0.5 * (value_max - self.value_min) * (
                                    1 + math.cos(math.pi * t / self.cosine_t))
                        for value_max in value_max_values
                    ]

                return values

            def get_epoch_values(self, epoch: int):
                return self._get_value(epoch)

class CosineScheduler(Scheduler):
    """
    Cosine decay with warmup.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Modified from timm's implementation.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_name: str,
                 t_max: int,
                 value_min: float = 0.,
                 warmup_t=0,
                 const_t=0,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field=param_name, initialize=initialize)

        assert t_max > 0
        assert value_min >= 0
        assert warmup_t >= 0
        assert const_t >= 0

        self.cosine_t = t_max - warmup_t - const_t
        self.value_min = value_min
        self.warmup_t = warmup_t
        self.const_t = const_t

        if self.warmup_t:
            self.warmup_steps = [(v - value_min) / self.warmup_t for v in self.base_values]
            super().update_groups(self.value_min)
        else:
            self.warmup_steps = []

    def _get_value(self, t):
        if t < self.warmup_t:
            values = [self.value_min + t * s for s in self.warmup_steps]
        elif t < self.warmup_t + self.const_t:
            values = self.base_values
        else:
            t = t - self.warmup_t - self.const_t

            value_max_values = [v for v in self.base_values]

            values = [
                self.value_min + 0.5 * (value_max - self.value_min) * (1 + math.cos(math.pi * t / self.cosine_t))
                for value_max in value_max_values
            ]

        return values

    def get_epoch_values(self, epoch: int):
        return self._get_value(epoch)

def pad_img(x, patch_size):
	_, _, h, w = x.size()
	mod_pad_h = (patch_size - h % patch_size) % patch_size
	mod_pad_w = (patch_size - w % patch_size) % patch_size
	x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
	return x