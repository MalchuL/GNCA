import torch.nn.functional as F
import numpy as np


def get_living_mask(x):
    alpha = x[:, :, :, 3:4]
    return F.max_pool2d(alpha, 3, 1, padding=1) > 0.1


def make_seed(size, n=1):
    x = np.zeros([n, size, size, CHANNEL_N], np.float32)
    x[:, size // 2, size // 2, 3:] = 1.0
    return x
