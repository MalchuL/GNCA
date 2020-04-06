import torch
import io
import PIL.Image, PIL.ImageDraw
import base64
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

def imread(path, max_size):
    img = plt.imread(path)
    img[:,:,:3] *= img[:,:,3:4]
    # 0: Nearest-neighbor 1: Bi-linear (default) 2: Bi-quadratic 3: Bi-cubic 4: Bi-quartic 5: Bi-quintic
    img = resize(img, (max_size,max_size), order=0, anti_aliasing=False)
    return img.transpose(2,0,1)


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return PIL.Image.fromarray(a)


def imwrite(f, a, fmt=None):
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit('.', 1)[-1].lower()
        if fmt == 'jpg':
            fmt = 'jpeg'
        f = open(f, 'wb')
    np2pil(a).save(f, fmt, quality=95)


def imencode(a, fmt='jpeg'):
    a = np.asarray(a)
    if len(a.shape) == 3 and a.shape[-1] == 4:
        fmt = 'png'
    f = io.BytesIO()
    imwrite(f, a, fmt)
    return f.getvalue()


def im2url(a, fmt='jpeg'):
    encoded = imencode(a, fmt)
    base64_byte_string = base64.b64encode(encoded).decode('ascii')
    return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string



def _to_NHWC(a):
    return a.transpose([0,2,3,1])

def _to_NCHW(a):
    return a.transpose([0,3,1,2])


def tile2d(a, w=None):

    a = np.asarray(a)

    a = _to_NHWC(a)
    if w is None:
        w = int(np.ceil(np.sqrt(len(a))))
    th, tw = a.shape[1:3]
    pad = (w - len(a)) % w
    a = np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), 'constant')
    h = len(a) // w
    a = a.reshape([h, w] + list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th * h, tw * w] + list(a.shape[4:]))

    return a


def zoom(img, scale=4):
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


def to_rgba(x):
    return x[:, :4, ...]


def to_alpha(x):
    return np.clip(x[:, 3:4, ...], 0.0, 1.0)


def to_rgb(x):
    # assume rgb premultiplied by alpha
    rgb, a = x[:, :3, ...], to_alpha(x)
    return 1.0 - a + rgb

