"""BALD information gain from aligned Monte Carlo dropout probabilities."""
from __future__ import annotations
import numpy as np
from .base import QueryBatch, QueryContext, integer, probability_array, ranked_batch
from .entropy import entropy_scores


def bald_scores(mc_probabilities: np.ndarray) -> np.ndarray:
    p = probability_array(mc_probabilities, ndim=3)
    if p.shape[0] < 2:
        raise ValueError("BALD requires at least two stochastic passes")
    score = entropy_scores(p.mean(axis=0)) - np.stack([entropy_scores(a) for a in p]).mean(axis=0)
    return np.maximum(score, 0)  # Remove roundoff below zero, not substantive signal.


class BALDAcquisition:
    def select(self, context: QueryContext) -> QueryBatch:
        if not context.effective_budget:
            return QueryBatch((), (), method="bald", pool_version=context.pool_version)
        if context.mc_probabilities is None:
            raise ValueError("BALD requires MC probabilities in current unlabeled ID order")
        return ranked_batch(context, bald_scores(context.mc_probabilities), "bald",
                            {"passes": context.mc_probabilities.shape[0],
                             "normalization": "log(8)"})


def mc_dropout_probabilities(model, inputs, *, passes: int = 10, seed: int = 42) -> np.ndarray:
    """Sample only dropout, freeze BatchNorm, and restore module modes/RNG states.

    Inputs are one already prepared tensor batch. No dataset or targets are accepted.
    A standard dropout-free ResNet must use a configured dropout head first.
    """
    import torch
    integer(passes, "passes", 2)
    integer(seed, "seed")
    dropout_types = (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d,
                     torch.nn.Dropout3d, torch.nn.AlphaDropout, torch.nn.FeatureAlphaDropout)
    dropout = [m for m in model.modules() if isinstance(m, dropout_types) and m.p > 0]
    if not dropout:
        raise ValueError("BALD requires a classifier with active dropout")
    modes = [(m, m.training) for m in model.modules()]
    cuda_devices = [inputs.device.index if inputs.device.index is not None else torch.cuda.current_device()] if inputs.is_cuda else []
    try:
        model.eval()
        for layer in dropout:
            layer.train()
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.random.default_generator.manual_seed(seed)
            for device in cuda_devices:
                torch.cuda.default_generators[device].manual_seed(seed)
            return np.stack([torch.softmax(model(inputs), dim=-1).cpu().numpy()
                             for _ in range(passes)])
    finally:
        for module, training in modes:
            module.training = training
