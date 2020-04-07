import pickle
import os
import numpy as np
import random
import torch
import torch.optim.lr_scheduler as sched
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from ..model.simple_model import CAModel
from ..training.pool import SamplePool, make_circle_masks, generate_pool_figures, visualize_batch
from ..utils import to_rgba, imwrite, to_rgb
from ..utils.ca_utils import clip_grad_norm_


def train(target_image, train_config, log_config, infer_config, use_cuda=False, resume=True):
    def move_(value):
        if use_cuda:
            return value.cuda()
        else:
            return value

    def infer_model(model, seed, step, output_folder):
        output_folder = Path(output_folder) / ('infer_' + str(step))
        output_folder.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            x = seed[None, ...]
            x = move_(torch.from_numpy(x))
            for i in range(infer_config.INFER_STEPS + 1):
                from src.utils.img_utils import _to_NHWC
                temp = _to_NHWC(to_rgb(x.detach().cpu().numpy()))[0]
                imwrite(os.path.join(output_folder, 'out_%04d.jpg' % i), temp)

                x = model(x)

    root_folder = Path(log_config.OUTPUT_FOLDER)

    train_output_folder = root_folder / 'train_log'
    train_output_folder.mkdir(parents=True, exist_ok=True)

    infer_output_folder = root_folder / 'infer_log'
    infer_output_folder.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(root_folder / 'metrics')

    p = train_config.TARGET_PADDING
    pad_target = np.pad(target_image, [(0, 0), (p, p), (p, p)])
    h, w = pad_target.shape[1:]
    seed = np.zeros([train_config.CHANNEL_N, h, w], np.float32)
    seed[3:, h // 2, w // 2] = 1.0
    pad_target = pad_target[None, ...]

    pad_target = move_(torch.from_numpy(pad_target))

    def loss_f(x, target):
        x = to_rgba(x)
        target = target.expand_as(x)
        loss = F.mse_loss(x, target, reduction='none')
        return loss.mean(-1).mean(-1).mean(-1)

    ca = move_(CAModel(train_config.CHANNEL_N, train_config.CELL_FIRE_RATE, 128))

    loss_log = []

    lr = train_config.LR

    optimizer = optim.Adam(ca.parameters(), lr)

    lr_sched = sched.MultiStepLR(optimizer,
                                 train_config.STEPS, gamma=0.1)

    try:
        if not resume:
            raise Exception("You forced --nor-resume flag, breaking loading")
        ckpt = torch.load(str(root_folder / 'model_last.ckpt'))
        ca.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_sched.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
    except Exception as e:
        print('Load error:', e)
        epoch = 0

    try:
        with open('pool.pkl', 'rb') as f:
            pool = pickle.load(f)
    except:
        pool = SamplePool(x=np.repeat(seed[None, ...], train_config.POOL_SIZE, 0))

    def train_step(x):
        optimizer.zero_grad()

        x = move_(torch.from_numpy(x))
        iter_n = int(random.uniform(*train_config.ITER_NUMBER))
        for i in range(iter_n):
            x = ca(x)
        loss = torch.mean(loss_f(x, pad_target))
        loss.backward()

        clip_grad_norm_(ca.parameters(), 1)

        optimizer.step()
        lr_sched.step()
        return x, loss

    for i in range(epoch, train_config.EPOCHES):
        if train_config.USE_PATTERN_POOL:
            batch = pool.sample(train_config.BATCH_SIZE)

            x0 = move_(torch.from_numpy(batch.x))
            loss_rank = loss_f(x0, pad_target).cpu().numpy().argsort()[::-1]
            x0 = x0.cpu().numpy()
            x0 = x0[loss_rank]

            seeded = train_config.BATCH_SIZE // 3

            x0[:seeded] = np.repeat(seed[None, ...], seeded, 0)
            if train_config.DAMAGE_N:
                damage = 1.0 - make_circle_masks(train_config.DAMAGE_N, h, w)[:, None, ...]
                x0[-train_config.DAMAGE_N:] *= damage
        else:
            batch = pool.sample(train_config.BATCH_SIZE)
            x0 = np.repeat(seed[None, ...].copy(), train_config.BATCH_SIZE, 0)

        x, loss = train_step(x0)

        batch.x[:] = x.detach().cpu().numpy()
        batch.commit()

        step_i = i
        loss = loss.item()
        loss_log.append(loss)

        if step_i % 10 == 0:
            generate_pool_figures(pool, step_i, train_output_folder)
            writer.flush()

        if step_i % 100 == 0:
            visualize_batch(x0, x.cpu().detach().numpy(), step_i, train_output_folder)

            ckpt = torch.save(
                dict(model=ca.state_dict(), optimizer=optimizer.state_dict(), scheduler=lr_sched.state_dict(), epoch=i),
                str(root_folder / 'model_last.ckpt'))

        if (step_i + 1) % infer_config.INFER_ON_EACH == 0:
            infer_model(ca, seed, step_i, infer_output_folder)

        print('\r step: %d, log10(loss): %.3f, loss: %f' % (i, np.log10(loss), loss), end='')
        # Metrics
        writer.add_scalar('lr', lr_sched.get_lr()[0], step_i)
        writer.add_scalar('MSE', loss, step_i)
        writer.add_scalar('log_MSE', np.log10(loss), step_i)

    infer_model(ca, seed, 'last', infer_output_folder)
