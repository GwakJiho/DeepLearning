import os
import yaml
import numpy as np
import random

CONFIG_PATH = 'config/'

def smooth_curve(x):  #사용 x
    window_len = 101
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[50:len(y)-50]


def shuffle_dataset(x, t):  #사용 보류
    shuffled = list(zip(x, t))
    random.shuffle(shuffled)
    x = [e[0] for e in shuffled]
    t = [e[0] for e in shuffled]

    return x, t

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config