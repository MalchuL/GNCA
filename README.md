# Growing Neural Cellar Automata

This repository is reimplementation on PyTorch code from this [blog](https://distill.pub/2020/growing-ca/)


Example of infer:\
     ![Image of Sonic](examples/sonic.gif)

## Installation
All experiments was on running **Ubuntu OS**, **NVDIA 2080 TI GPU**.

Requirements:

    * Python 3.6+
    * CUDA 9.1+
    * 10GB GPU Memory 
   
    
Steps to reproduce:

    1. Run `pip install -r requirements.txt`.   
    2. Run `python main.py` to use CPU or  `python main.py --use-cuda` to use GPU
    3. Wait while training will ends, in infer_log folder will be generated image  
    
