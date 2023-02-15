from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    The customized Learning Rate Scheduler:

    Arguments:
        _LRScheduler: lr scheduler

    """

    def __init__(self, optimizer, num_of_epoch, ini_lr, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.
        Arguments:
            num_of_epoch: number of epoch
            init_lr: initialized learning rate
        """
        # ... Your Code Here ...
        self.num_of_epoch = num_of_epoch
        self.ini_lr = ini_lr
        self.decay = [self.ini_lr / i for i in range(1, num_of_epoch + 1, 1)]
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Return:
            the learning rate for each epoch
        """

        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        return [
            self.ini_lr / (1 + self.decay[i - 1] * i)
            for i in range(1, self.num_of_epoch + 1, 1)
        ]
