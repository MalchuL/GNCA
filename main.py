import numpy as np

import src.utils as utils
from src.training.train import train
from src.utils import imread
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser('Growing Neural Cellular Automata for single image restoration')
parser.add_argument('--config', default='configs/config.yml', help='path to config')
parser.add_argument('--use-cuda', action='store_true', help='use cuda?')
parser.add_argument('--skip-preview', action='store_true', help='Skip image preview')
parser.add_argument('--not-resume', action='store_true', help='resume training (default = True)')

if __name__ == '__main__':
    args = parser.parse_args()

    yaml = utils.load_as_object(args.config)

    img = imread(yaml.data.IMG_PATH, yaml.model.TARGET_SIZE)

    if not args.skip_preview:
        plt.imshow(img.transpose(1, 2, 0))
        plt.show()

    args.resume = not args.not_resume
    train(img.astype(np.float32), yaml.model, yaml.log, yaml.infer, args.use_cuda, args.resume)
