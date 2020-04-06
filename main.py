import numpy as np

from src import *
from src.train.train import main
from src.utils import imread
import matplotlib.pyplot as plt

if __name__ == '__main__':
    yaml = utils.load_as_object('configs/config.yml')
    print(yaml.model)

    img = imread(yaml.data.IMG_PATH, yaml.model.TARGET_SIZE)

    plt.imshow(img.transpose(1,2,0))
    plt.show()

    print(img.shape)
    main(img.astype(np.float32), yaml.model, yaml.log)



