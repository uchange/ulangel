import torch
import torch.nn.functional as F

from ulangel.utils.callbacks import Callback, listify


class AvgStats:
    """This class AvgStats is a recorder of all performance statistics. It
    records the loss value in self.tot_los and values calculated by the input
    performance metrics in self.tot_mets. It counts and saves the number of
    calculated exemples in self.count. The method accumulate is to sum all
    statistics after running a batch and to update the count value. It can also
    print average statistics depending on the training or validation state.
    """

    def __init__(self, metrics, in_train):
        self.metrics = listify(metrics)
        self.in_train = in_train

    def reset(self):
        self.tot_loss = 0.0
        self.count = 0
        self.tot_mets = [0.0] * len(self.metrics)

    @property
    def all_stats(self):
        "return the total of all statistic: loss + values given by metrics"
        return [self.tot_loss.item()] + self.tot_mets

    @property
    def avg_stats(self):
        "return the average of all statistic: loss + values given by metrics"
        return [o / self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count:
            return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        "update statistics and count"
        bn = run.yb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn


class AvgStatsCallback(Callback):
    """Calculating all statistics after every batch and print all
    statitics after every epoch.
    """

    def __init__(self, metrics):

        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)


# accuracy metric for classifier
def accuracy(input, targs):
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n, -1)
    targs = targs.view(n, -1)
    return (input == targs).float().mean()


# accuracy_flat metric for language model
def accuracy_flat(input, target):
    "flating a batch of outputs of a language model then calculating the accuracy"
    bs, sl = target.size()
    return accuracy(input.view(bs * sl, -1), target.view(bs * sl))


# cross_entropy_flat is not a metric. It is a loss function, but it has the same
# pricipal as accuracy_flat, so I put it here.
def cross_entropy_flat(input, target):
    bs, sl = target.size()
    return F.cross_entropy(input.view(bs * sl, -1), target.view(bs * sl))
