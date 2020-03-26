import numpy as np
import random
import torch
import torch.optim.lr_scheduler as sched
import torch.optim as optim
import torch.nn.functional as F

from src.model.simple_model import CAModel
from src.train.pool import SamplePool, make_circle_masks, generate_pool_figures, visualize_batch
from src.utils import to_rgba


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





def main(target_image, train_config):
    p = train_config.TARGET_PADDING
    print(type(p), target_image.shape)
    pad_target = np.pad(target_image, [(0, 0), (p, p), (p, p)])
    h, w = pad_target.shape[1:]
    seed = np.zeros([train_config.CHANNEL_N, h, w], np.float32)
    seed[3:, h // 2, w // 2] = 1.0
    pad_target = pad_target[None, ...]

    pad_target = torch.from_numpy(pad_target).cuda()


    def loss_f(x, target):
        x = to_rgba(x)
        target = target.expand_as(x)
        loss = F.mse_loss(x, target, reduction='none')
        return loss.mean(-1).mean(-1).mean(-1)

    ca = CAModel(train_config.CHANNEL_N, train_config.CELL_FIRE_RATE).cuda()

    loss_log = []

    lr = 2e-3

    optimizer = optim.Adam(ca.parameters(), lr)

    lr_sched = sched.MultiStepLR(optimizer,
                                 [5000, 8000], gamma=0.1)

    pool = SamplePool(x=np.repeat(seed[None, ...], train_config.POOL_SIZE, 0))

    def train_step(x):
        optimizer.zero_grad()

        x = torch.from_numpy(x).cuda()
        iter_n = int(random.uniform(64, 96))
        for i in range(iter_n):
            x = ca(x)
        loss = torch.mean(loss_f(x, pad_target))
        loss.backward()

        clip_grad_norm_(ca.parameters(), 1)

        optimizer.step()
        lr_sched.step()
        return x, loss

    for i in range(10000 + 1):
        if train_config.USE_PATTERN_POOL:
            batch = pool.sample(train_config.BATCH_SIZE)

            x0 = torch.from_numpy(batch.x).cuda()
            loss_rank = loss_f(x0, pad_target).cpu().numpy().argsort()[::-1]
            x0 = x0.cpu().numpy()
            x0 = x0[loss_rank]

            seeded = train_config.BATCH_SIZE // 3

            x0[:seeded] = np.repeat(seed[None, ...], seeded, 0)
            if train_config.DAMAGE_N:
                damage = 1.0 - make_circle_masks(train_config.DAMAGE_N, h, w)[:,None,...]
                x0[-train_config.DAMAGE_N:] *= damage
        else:
            batch = pool.sample(train_config.BATCH_SIZE)
            x0 = np.repeat(seed[None, ...].copy(), train_config.BATCH_SIZE, 0)


        x, loss = train_step(x0)


        batch.x[:] = x.detach().cpu().numpy()
        batch.commit()

        step_i = len(loss_log)
        loss = loss.item()
        loss_log.append(loss)

        #import matplotlib.pyplot as plt
        if step_i % 10 == 0:
            generate_pool_figures(pool, step_i)

        if step_i % 100 == 0:
            visualize_batch(x0, x.cpu().detach().numpy(), step_i)



        print('\r step: %d, %f log10(loss): %.3f' % (len(loss_log), loss, np.log10(loss)), end='')
