import torch.nn.functional as F
import numpy as np


def get_living_mask(x):
    alpha = x[:, 3:4, :, :]
    return F.max_pool2d(alpha, 3, 1, padding=1) > 0.1

