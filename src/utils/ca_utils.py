import torch.nn.functional as F
import numpy as np
import torch

def get_living_mask(x):
    alpha = x[:, 3:4, :, :]
    return F.max_pool2d(alpha, 3, 1, padding=1) > 0.1



def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        clip_coef = max_norm / (param_norm + 1e-8)
        p.grad.data.mul_(clip_coef)



