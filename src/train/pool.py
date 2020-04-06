import numpy as np
import os
from src.utils import tile2d, to_rgb, imwrite
from src.utils.img_utils import _to_NHWC


class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = list(slots.keys())
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k).copy()


def make_circle_masks(n, h, w):
    x = np.linspace(-1.0, 1.0, w)[None, None, :]
    y = np.linspace(-1.0, 1.0, h)[None, :, None]
    center = np.random.uniform(-0.5, 0.5, [2, n, 1, 1])
    r = np.random.uniform(0.1, 0.4, [n, 1, 1])
    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = (x * x + y * y < 1.0).astype(np.float32)
    return mask


def export_model(ca, base_fn):
    raise NotImplemented


def generate_pool_figures(pool, step_i, output_folder):
    tiled_pool = tile2d(to_rgb(pool.x[:49]))
    fade = np.linspace(1.0, 0.0, 72)
    ones = np.ones(72)
    tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None]
    tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
    tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
    tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]
    imwrite(os.path.join(output_folder, '%04d_pool.jpg' % step_i), tiled_pool)


def visualize_batch(x0, x, step_i, output_folder):
    vis0 = np.hstack(_to_NHWC(to_rgb(x0)))
    vis1 = np.hstack(_to_NHWC(to_rgb(x)))
    vis = np.vstack([vis0, vis1])
    imwrite(os.path.join(output_folder, 'batches_%04d.jpg' % step_i), vis)
