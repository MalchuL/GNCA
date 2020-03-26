import numpy as np

from src import *
from src.train.train import main
from src.utils import imread, load_image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    yaml = utils.load_as_object('configs/config.yml')
    print(yaml.model)

    img = load_image(yaml.data.IMG_PATH, yaml.model.TARGET_SIZE)
    print(img.shape)
    main(img.astype(np.float32), yaml.model)



