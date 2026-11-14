"""MC Dropout controls and population statistics, separate from display scaling."""

from contextlib import contextmanager
import torch
from torch import nn


def active_dropout_layers(*models):
    return {
        f"model_{index}.{name}": module.p
        for index, model in enumerate(models)
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.p > 0
    }


@contextmanager
def mc_dropout(*models):
    states = {module: module.training for model in models for module in model.modules()}
    active = active_dropout_layers(*models)
    if not active:
        raise ValueError("MC Dropout requires nonzero dropout trained with the model; this checkpoint has none.")
    try:
        for model in models:
            model.eval()
        for module in states:
            if isinstance(module, nn.modules.dropout._DropoutNd) and module.p > 0:
                module.train()
        yield active
    finally:
        for module, training in states.items():
            module.training = training


def ensemble_statistics(predictions):
    if predictions.shape[0] < 2:
        raise ValueError("At least two MC samples are required.")
    # Paper equation uses 1/R, rather than the default 1/(R-1).
    return predictions.mean(dim=0), predictions.var(dim=0, unbiased=False)
