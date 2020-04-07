import numpy as np


from src import *
from src.training.train import train
from src.utils import imread
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser('Growing Neural Cellar Automata for single image restoration')
parser.add_argument('--config', default='configs/config.yml', help='path to config')

if __name__ == '__main__':
    args = parser.parse_args()

    yaml = utils.load_as_object(args.config)

    img = imread(yaml.data.IMG_PATH, yaml.model.TARGET_SIZE)

    plt.imshow(img.transpose(1, 2, 0))
    plt.show()

    train(img.astype(np.float32), yaml.model, yaml.log)
