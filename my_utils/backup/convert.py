import numpy as np
from PIL import Image
import torch

def srgb_to_linear(srgb):
    '''
    Convert sRGB [0, 1] to linear [0, +inf)
    '''
    a = 0.055
    # srgb is on tensor
    return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1 + a))**2.4)
    
def linear_to_srgb(linear):
    '''
    convert linear [0, +inf) to sRGB [0, 1]
    '''
    a = 0.055
    return torch.where(linear <= 0.0031308, 12.92 * linear, (1 + a) * linear**(1 / 2.4) - a)

    

