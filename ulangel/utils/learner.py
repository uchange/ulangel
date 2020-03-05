import torch

from ulangel.utils.callbacks import (
    CancelBatchException,
    CancelEpochException,
    CancelTrainException,
    listify,
)


class Learner:
    """An object to use the defined RNN model, the databunch, all callbacks,
    the loss function ,the optimization function to train.
    """

    def __init__(
        self, model, data, loss_func, opt_func, lr=1e-2, cbs=None, cb_funcs=None
    ):
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.lr = lr
        self.in_train = False
        self.logger = print
        self.opt = None

        # NB: Things marked "NEW" are covered in lesson 12
        # NEW: avoid need for set_runner
        self.cbs = []
        # self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cb_funcs))

    def add_cbs(self, cbs):
        for cb in listify(cbs):
            self.add_cb(cb)

    def add_cb(self, cb):
        cb.set_runner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        for cb in listify(cbs):
            self.cbs.remove(cb)

    def one_batch(self, i, xb, yb):
        try:
            self.iter = i
            self.xb = xb
            self.yb = yb

            # I added to debug from our data
            self.yb = self.yb.long()
            self("begin_batch")

            self.pred = self.model(self.xb)
            self("after_pred")
            self.loss = self.loss_func(self.pred, self.yb)
            self("after_loss")
            if not self.in_train:
                return
            self.loss.backward()
            self("after_backward")
            self.opt.step()
            self("after_step")
            self.opt.zero_grad()
        except CancelBatchException:
            self("after_cancel_batch")
        finally:
            self("after_batch")

    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for i, (xb, yb) in enumerate(self.dl):
                self.one_batch(i, xb, yb)
        except CancelEpochException:
            self("after_cancel_epoch")

    def do_begin_fit(self, epochs):
        self.epochs = epochs
        self.loss = torch.as_tensor(0.0)
        self("begin_fit")

    def do_begin_epoch(self, epoch):
        self.epoch = epoch
        self.dl = self.data.train_dl
        print(self.epoch)
        return self("begin_epoch")

    def fit(self, epochs, cbs=None, reset_opt=False):
        # NEW: pass callbacks to fit() and have them removed when done
        self.add_cbs(cbs)
        # NEW: create optimizer on fit(), optionally replacing existing
        if reset_opt or not self.opt:
            self.opt = self.opt_func(self.model.parameters(), lr=self.lr)

        try:
            self.do_begin_fit(epochs)
            for epoch in range(epochs):
                self.do_begin_epoch(epoch)
                if not self("begin_epoch"):
                    self.all_batches()

                with torch.no_grad():
                    self.dl = self.data.valid_dl
                    if not self("begin_validate"):
                        self.all_batches()
                self("after_epoch")

        except CancelTrainException:
            self("after_cancel_train")
        finally:
            self("after_fit")
            self.remove_cbs(cbs)

    ALL_CBS = {
        "begin_batch",
        "after_pred",
        "after_loss",
        "after_backward",
        "after_step",
        "after_cancel_batch",
        "after_batch",
        "after_cancel_epoch",
        "begin_fit",
        "begin_epoch",
        "begin_validate",
        "after_epoch",
        "after_cancel_train",
        "after_fit",
    }

    def __call__(self, cb_name):
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) and res
        return res


def freeze_all(model_all_layers):
    for layer in model_all_layers:
        for operation in layer:
            for param in operation.parameters():
                param.requires_grad = False


def unfreeze_all(model_all_layers):
    for layer in model_all_layers:
        for operation in layer:
            for param in operation.parameters():
                param.requires_grad = True


def freeze_upto(model_all_layers, nb_layer):
    freeze_all(model_all_layers)
    unfreeze_layers = model_all_layers[nb_layer:]
    for layer in unfreeze_layers:
        for operation in layer:
            for param in operation.parameters():
                param.requires_grad = True
