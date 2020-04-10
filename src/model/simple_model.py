import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from ..utils.ca_utils import get_living_mask

def ca_block(in_size, out_size):
    return nn.Sequential(nn.Conv2d(in_size, out_size, 1), nn.ReLU6(inplace=True))

class CAModel(nn.Module):

    def __init__(self, channel_n, fire_rate, hidden_sizes=(128,)):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate


        layers = []
        sizes = zip([channel_n * 3] + hidden_sizes[:-1], hidden_sizes)
        for in_size, out_size in sizes:
            layers.append(ca_block(in_size, out_size))


        self.dmodel = nn.Sequential(*layers, nn.Conv2d(hidden_sizes[-1], channel_n, 1))

        self._init_weights()

    def _get_kernel(self, angle=0.0):
        identify = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T.copy()
        c, s = np.cos(angle), np.sin(angle)

        identify = np.repeat(identify[None, :, :], self.channel_n, 0)

        x = c * dx - s * dy
        x = np.repeat(x[None, :, :], self.channel_n, 0)

        y = s * dx + c * dy
        y = np.repeat(y[None, :, :], self.channel_n, 0)

        kernel = np.concatenate([identify, x, y], 0)[:, None, :, :].astype(np.float32)

        kernel = torch.from_numpy(kernel)
        kernel.requires_grad = False
        return kernel




    def _init_weights(self):
        nn.init.zeros_(self.dmodel[-1].weight)
        nn.init.zeros_(self.dmodel[-1].bias)

    def perceive(self, x, angle=0.0):
        kernel = self._get_kernel(angle)
        kernel = kernel.to(x.device)  # Move to same device

        x = x.repeat(1,3,1,1)
        channels_count = x.size()[1]  # NHWC correspondence
        y = F.conv2d(x, kernel, padding=1, groups=channels_count)
        return y

    def forward(self, x):
        return self.call(x)

    def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
        pre_life_mask = get_living_mask(x)

        y = self.perceive(x, angle)
        dx = self.dmodel(y) * step_size

        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = torch.rand(x[:, :1, :, :].shape, device=x.device) <= fire_rate
        update_mask.float().to(x.device)
        update_mask.requires_grad = False

        x = x + dx * update_mask

        post_life_mask = get_living_mask(x)
        life_mask = (pre_life_mask & post_life_mask).float().to(x.device)
        life_mask.requires_grad = False
        return x * life_mask
