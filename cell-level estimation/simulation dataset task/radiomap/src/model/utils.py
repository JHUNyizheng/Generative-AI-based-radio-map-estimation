import os
import cv2
import numpy as np

import sys
import time
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, average_precision_score
from sklearn.metrics import mean_squared_error, r2_score
# noinspection PyProtectedMember
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.detach().cpu().numpy()
    # y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)
    score_average_precision = average_precision_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_average_precision]

def RMSE(result,test_y):
    return np.sqrt(mean_squared_error(result, test_y))

def calculate_rmse(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_true = y_true.reshape(-1)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1)
    """ Prediction """
    score_RMSE = RMSE(y_pred, y_true)
    return score_RMSE

# noinspection PyAttributeOutsideInit
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: init learning rate = base lr / multiplier
          warmup_epoch: target learning rate is reached at warmup_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step(epoch - self.warmup_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        state = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)


def get_scheduler(optimizer, n_iter_per_epoch, args):
    warmup_epoch = 5
    warmup_multiplier = 10
    if args.arch == 'RadioCycle':
        optimizer = optimizer[1]
    else:
        optimizer = optimizer[0]
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        eta_min=0.000001,
        T_max=(args.num_epochs - warmup_epoch) * n_iter_per_epoch)

    if warmup_epoch > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=warmup_multiplier,
            after_scheduler=scheduler,
            warmup_epoch=warmup_epoch * n_iter_per_epoch)
    return scheduler

