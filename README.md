# Growing Neural Cellar Automata

This repository is reimplementation on PyTorch code from this [blog](https://distill.pub/2020/growing-ca/)


Example of infer:\
     ![Image of Sonic](examples/sonic.gif)

## Installation
All experiments was on running **Ubuntu OS**, **NVDIA 2080 TI GPU**.

Requirements:

    * Python 3.6+
    * **16GB RAM Memory** (Please be carefully when you run script)
    * CUDA 9.1+ (For GPU training)
    * 10GB GPU Memory (For GPU training) 
   
    
## Steps to reproduce:

1. Run `pip install -r requirements.txt`.   
2. Run `python main.py` to use CPU or  `python main.py --use-cuda` to use GPU
3. Wait while training will ends, in infer_log folder will be generated image  

## How to decrease Memory usage:
* For GPU(with --use-cuda parameter) or CPU(w/o --use-cuda parameter) you can change next parameters in config:
    1. decrease both values in ITER_NUMBER tuple
    2. decrease BATCH_SIZE
    3. decrease TARGET_SIZE
    
* For CPU only usage:
    1. decrease POOL_SIZE 
## Make GIF

0. (Optional) Run for resize in <images_folder> folder `for X in *; do convert $X -interpolate Nearest -filter point -resize 480x480 $X; done`
1. To make GIF run `convert -delay 20 -loop 0 <images_folder>/*.jpg myimage.gif` 