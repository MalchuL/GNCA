import numpy as np
import random
import torch
import torch.optim.lr_scheduler as sched
import torch.optim as optim
import torch.nn.functional as F

from src.model.simple_model import CAModel
from src.train.pool import SamplePool, make_circle_masks, generate_pool_figures
from src.utils import to_rgba


def main(target_image, train_config):
    p = train_config.TARGET_PADDING
    print(type(p), target_image.shape)
    pad_target = np.pad(target_image, [(0, 0), (p, p), (p, p)])
    h, w = pad_target.shape[1:]
    seed = np.zeros([train_config.CHANNEL_N, h, w], np.float32)
    seed[3:, h // 2, w // 2] = 1.0
    pad_target = pad_target[None, ...]

    pad_target = torch.from_numpy(pad_target)
    target_image = torch.from_numpy(target_image)


    def loss_f(x, target):
        x =  to_rgba(x)
        target = target.expand_as(x)
        loss = F.mse_loss(to_rgba(x), target, reduction='none')
        return loss.mean(-1).mean(-1).mean(-1)

    ca = CAModel(train_config.CHANNEL_N, train_config.CELL_FIRE_RATE)

    loss_log = []

    lr = 2e-3

    optimizer = optim.Adam(ca.parameters(), lr)

    lr_sched = sched.MultiStepLR(optimizer,
                                 [2000], gamma=0.1)

    pool = SamplePool(x=np.repeat(seed[None, ...], train_config.POOL_SIZE, 0))

    def train_step(x):
        x = torch.from_numpy(x)
        iter_n = int(random.uniform(46, 72))
        for i in range(iter_n):
            x = ca(x)
        loss = torch.mean(loss_f(x, pad_target))
        loss.backward()
        optimizer.step()
        return x, loss

    for i in range(8000 + 1):
        if train_config.USE_PATTERN_POOL:
            batch = pool.sample(train_config.BATCH_SIZE)

            x0 = torch.from_numpy(batch.x)
            loss_rank = loss_f(x0, pad_target).numpy().argsort()[::-1]
            print(loss_rank)
            x0 = x0.numpy()
            x0 = x0[loss_rank]
            x0[:1] = seed
            if train_config.DAMAGE_N:
                print(type(train_config.DAMAGE_N))
                damage = 1.0 - make_circle_masks(train_config.DAMAGE_N, h, w)[:,None,...]
                print(x0.shape, damage.shape)
                x0[-train_config.DAMAGE_N:] *= damage
        else:
            x0 = np.repeat(seed[None, ...], train_config.BATCH_SIZE, 0)

        x, loss = train_step(x0)

        if train_config.USE_PATTERN_POOL:
            batch.x[:] = x.detach().cpu().numpy()
            batch.commit()

        step_i = len(loss_log)
        loss = loss.item()
        loss_log.append(loss)


        if step_i % 10 == 0:
            generate_pool_figures(pool, step_i)
        if False and step_i % 100 == 0:
            clear_output()
            visualize_batch(x0, x, step_i)
            plot_loss(loss_log)
            export_model(ca, 'train_log/%04d' % step_i)

        print('\r step: %d, log10(loss): %.3f' % (len(loss_log), np.log10(loss)), end='')
