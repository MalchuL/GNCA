import numpy as np

from src import *
from src.train.train import main
from src.utils import imread

if __name__ == '__main__':
    yaml = utils.load_as_object('configs/config.yml')
    print(yaml.model)

    img = imread(yaml.data.IMG_PATH)
    print(img.shape)

    main(img.astype(np.float32), yaml.model)



