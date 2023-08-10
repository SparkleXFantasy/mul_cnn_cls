import numpy as np
import os
import random
import torch
from matplotlib import pyplot as plt
from torch import nn

def cls2one_hot(cls, shape):
    try:
        assert(len(shape) == 2)
    except:
        print('The expected shape of a one hot vector should be 2, got {}.'.format(len(shape)))
    one_hot = torch.zeros(shape)
    one_hot[:, cls] = 1
    return one_hot


def init_layer(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def set_random_seed(seed_value):
    seed_value = seed_value
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True


def grayscale_to_color(gray_img, colormap='jet'):
    # Normalize grayscale image from numpy to [0, 1]
    eps = 1e-8
    norm_img = (gray_img - np.min(gray_img)) / (np.max(gray_img) - np.min(gray_img) + eps)

    # Use colormap to create color image
    cmap = plt.get_cmap(colormap)
    color_img = cmap(norm_img)[:, :, :3]  # Keep only RGB channels, ignore alpha channel if present

    return (color_img * 255).astype(np.uint8)