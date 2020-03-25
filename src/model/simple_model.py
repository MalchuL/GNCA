import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from src.utils.ca_utils import get_living_mask


class CAModel(nn.Module):

    def __init__(self, channel_n, fire_rate, hidden_size=128):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        self.hidden_size = hidden_size

        self.dmodel = nn.Sequential(
            nn.Conv2d(self.channel_n, self.hidden_size, 1),
            nn.ReLU()
        )
        self.out = nn.Conv2d(self.hidden_size, self.channel_n, 1)

        self._init_weights()


    def _get_kernel(self, angle=0.0):
        identify = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T.copy()
        c, s = np.cos(angle), np.sin(angle)
        kernel = np.stack([identify, c * dx - s * dy, s * dx + c * dy], 0)[:,None, :, :].astype(np.float32)
        kernel = np.repeat(kernel, self.channel_n, 1)
        kernel = kernel.transpose([1, 0, 2, 3])
        kernel = torch.from_numpy(kernel)
        kernel.requires_grad = False
        return kernel




    def _init_weights(self):
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def perceive(self, x, angle=0.0):
        kernel = self._get_kernel(angle)
        kernel = kernel.to(x.device)  # Move to same device

        channels_count = x.size()[1] # NHWC correspondence

        x = x.repeat(1,3,1,1)
        y = F.conv2d(x, kernel, padding=1, groups=channels_count)
        return y

    def forward(self, x):
        x = self.perceive(x)
        dx = self.dmodel(x)
        dx = self.out(dx)
        return dx

    def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
        pre_life_mask = get_living_mask(x)

        y = self.perceive(x, angle)
        dx = self.dmodel(y) * step_size
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
        x += dx * tf.cast(update_mask, tf.float32)

        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        return x * tf.cast(life_mask, tf.float32)
