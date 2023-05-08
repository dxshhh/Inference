import numpy as np
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
from main import sample, PCN, show_2D, plot_neuron_2
import matplotlib.image as mpimg
from PIL import Image
import moviepy.editor as mp
from matplotlib.animation import ArtistAnimation

if __name__ == '__main__':
  a = np.ones((2,2))
  b = np.array([a])
  print(b.shape)


