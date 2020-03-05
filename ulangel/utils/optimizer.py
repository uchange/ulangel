from functools import partial

import torch

from ulangel.utils.callbacks import listify


# some useful functions
def maybe_update(os, dest, f):
    "if k not in dest, update it's value dest[k] = v"
    for o in os:
        for key, value in f(o).items():
            if key not in dest:
                dest[key] = value


def get_defaults(d):
    return getattr(d, "_defaults", {})


def compose(x, funcs, *args, order_key="_order", **kwargs):
    "return steppers' result in the ascending way of steppers' order"
    # key = lambda o: getattr(o, order_key, 0)
    def key(obj):
        "get steppers' order"
        return getattr(obj, order_key, 0)

    for f in sorted(listify(funcs), key=key):
        x = f(x, **kwargs)
    return x


def debias(mom, damp, step):
    return damp * (1 - mom ** step) / (1 - mom)


# steppers
def sgd_step(p, lr, **kwargs):
    "basic stochastic gradient descent stepper"
    p.data.add_(-lr, p.grad.data)
    return p


def weight_decay(p, lr, wd, **kwargs):
    "weight decay stepper"
    p.data.mul_(1 - lr * wd)
    return p


# where to put this default setting?
weight_decay._defaults = dict(wd=0.0)


def adam_step(
    p, lr, mom, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, **kwargs
):
    "adam stepper"
    debias1 = debias(mom, mom_damp, step)
    debias2 = debias(sqr_mom, sqr_damp, step)
    p.data.addcdiv_(-lr / debias1, grad_avg, (sqr_avg / debias2).sqrt() + eps)
    return p


# where to put this default setting?
adam_step._defaults = dict(eps=1e-5)


# optimizer class
class Optimizer:
    def __init__(self, params, steppers, **defaults):
        self.steppers = listify(steppers)
        # if '_defaults' attribute of every stepper is not in defaults, add it
        maybe_update(self.steppers, defaults, get_defaults)
        # might be a generator
        self.param_groups = list(params)
        # ensure params is a list of lists
        if not isinstance(self.param_groups[0], list):
            self.param_groups = [self.param_groups]
        self.hypers = [{**defaults} for p in self.param_groups]

    def grad_params(self):
        "return param_groups requiring update"
        return [
            (p, hyper)
            for pg, hyper in zip(self.param_groups, self.hypers)
            for p in pg
            if p.grad is not None
        ]

    def zero_grad(self):
        for p, hyper in self.grad_params():
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        "apply stepper funciton on every parameters in the param_groups"
        for p, hyper in self.grad_params():
            compose(p, self.steppers, **hyper)


class StatefulOptimizer(Optimizer):
    "optimizer with states, which record the history parameter updates"

    def __init__(self, params, steppers, stateupdaters=None, **defaults):
        self.stateupdaters = listify(stateupdaters)
        maybe_update(self.stateupdaters, defaults, get_defaults)
        super().__init__(params, steppers, **defaults)
        self.state = {}

    def step(self):
        for p, hyper in self.grad_params():
            if p not in self.state:
                # Create a state for p and call all the statistics to initialize it.
                self.state[p] = {}
                # apply all initialization of self.stateupdaters on p and add the results into self.state[p]
                maybe_update(
                    self.stateupdaters, self.state[p], lambda o: o.init_state(p)
                )
            state = self.state[p]
            for stateupdater in self.stateupdaters:
                state = stateupdater.update(p, state, **hyper)
            compose(p, self.steppers, **state, **hyper)
            self.state[p] = state


# stat to calculate momentum
class StateUpdater:
    _defaults = {}

    def init_state(self, p):
        raise NotImplementedError

    def update(self, p, state, **kwargs):
        raise NotImplementedError


class AverageGrad(StateUpdater):
    "Momentum created by averaging the gradient"
    _defaults = dict(mom=0.9)

    def __init__(self, dampening: bool = False):
        self.dampening = dampening

    def init_state(self, p):
        return {"grad_avg": torch.zeros_like(p.grad.data)}

    def update(self, p, state, mom, **kwargs):
        state["mom_damp"] = 1 - mom if self.dampening else 1.0
        state["grad_avg"].mul_(mom).add_(state["mom_damp"], p.grad.data)
        return state


class AverageSqrGrad(StateUpdater):
    _defaults = dict(sqr_mom=0.99)

    def __init__(self, dampening: bool = True):
        self.dampening = dampening

    def init_state(self, p):
        return {"sqr_avg": torch.zeros_like(p.grad.data)}

    def update(self, p, state, sqr_mom, **kwargs):
        state["sqr_damp"] = 1 - sqr_mom if self.dampening else 1.0
        state["sqr_avg"].mul_(sqr_mom).addcmul_(
            state["sqr_damp"], p.grad.data, p.grad.data
        )
        return state


class StepCount(StateUpdater):
    def init_state(self, p):
        return {"step": 0}

    def update(self, p, state, **kwargs):
        state["step"] += 1
        return state


# inbuild optimizers
def adam_opt(xtra_step=None, **kwargs):
    "adam optimizer"
    return partial(
        StatefulOptimizer,
        steppers=[adam_step, weight_decay] + listify(xtra_step),
        stateupdaters=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()],
        **kwargs
    )


def sgd_opt():
    "stochastic gradient descent optimizer"
    return partial(Optimizer, steppers=[weight_decay, sgd_step])
