import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
import math
import jax
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2
import os
from matplotlib.animation import ArtistAnimation
from functools import partial
from PIL import Image


def load_config(model):
    normalize = transforms.Normalize(mean=[0], std=[1])
    inv_normalize = transforms.Normalize(mean=[-0 / 1], std=[1 / 1])
    _transforms = transforms.Compose([transforms.ToTensor(), normalize])
    if model == 0:
        neuron_size = [0, 200, 100, 50, 50]
        model_name = 'net.bp'

    elif model == 1:
        neuron_size = [0, 800, 400, 100]
        model_name = 'net'

    elif model == 2:
        neuron_size = [0, 200, 200, 200, 100, 100, 100, 50]
        model_name = '2net'
        training_data = datasets.FashionMNIST(
            root='iuput_data',
            train=True,
            download=True,
            transform=_transforms
        )
        using_epoch = 19

    elif model == 3:
        neuron_size = [0, 2000, 2000, 500]
        model_name = 'CIFAR10_net'
        training_data = datasets.CIFAR10(
            root='iuput_data',
            train=True,
            download=True,
            transform=_transforms
        )
        using_epoch = 3

    elif model == 4:
        neuron_size = [0, 500, 300, 400, 200, 100, 50, 25]
        model_name = 'Fashionmnistnet'
        training_data = datasets.FashionMNIST(
            root='iuput_data',
            train=True,
            download=True,
            transform=_transforms
        )
        using_epoch = 19
    return neuron_size,model_name,normalize,inv_normalize,_transforms,training_data,using_epoch