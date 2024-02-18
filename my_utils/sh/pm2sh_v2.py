'''
SH implementation reference: https://github.com/abdallahdib/NextFace/blob/main/sphericalharmonics.py
'''


import math
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import pyshtools as pysh 
import sys
import imageio
import OpenEXR


def associated_legendre_polynomial(l, m, x):
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def normlizeSH(l, m):
    return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / (4 * math.pi * math.factorial(l + m)))


def SH(l, m, theta, phi):
    if m == 0:
        return normlizeSH(l, m) * associated_legendre_polynomial(l, m, torch.cos(theta))
    elif m > 0:
        return math.sqrt(2.0) * normlizeSH(l, m) * \
                torch.cos(m * phi) * associated_legendre_polynomial(l, m, torch.cos(theta))
    else:
        return math.sqrt(2.0) * normlizeSH(l, -m) * \
                torch.sin(-m * phi) * associated_legendre_polynomial(l, -m, torch.cos(theta))

def get_sh_coeffs(direction=[0.,0.], order=3):
    '''
    input: direction [b,3]
    output: 
        coeffs: with size of [b,3,order**2]
    '''
    phi, theta = direction
    phi = phi + math.pi # from [-pi, pi] to [0, 2pi]
    phi, theta = torch.tensor(phi), torch.tensor(theta)

    sh_basis = []
    for l in range(order):
        for m in range(-l, l + 1):
            sh_basis.append(SH(l, m, theta, phi))
    sh_basis = torch.stack(sh_basis, dim=-1)  # [h,w,n]
    
    # 将系数扩展为所需的形状 [3, n]
    coeffs = sh_basis.unsqueeze(0).repeat(3, 1)
    
    # 将coeffs转化为float类型
    coeffs = coeffs.float()
    return coeffs

def get_pm_from_sh(coeffs, resolution=[32, 16], order=3):
    '''
    input: coeffs [b,3,order**2]
    output: 
        pm: [b,3,h,w] the env map represented by SH basis
    '''
    w, h = resolution
    
    theta = torch.linspace(0, math.pi, h)  # [h] from 0 to pi
    phi = torch.linspace(0, 2 * math.pi, w)  # [w] from 0 to 2pi
    theta = theta[..., None].repeat(1, w)  # [h,w]
    phi = phi[None, ...].repeat(h, 1)  # [h,w]

    sh_basis = []
    for l in range(order):
        for m in range(-l, l + 1):
            sh_basis.append(SH(l, m, theta, phi))
    sh_basis = torch.stack(sh_basis, dim=-1)  # [h,w,n]

    coeffs_ = coeffs[:, None, None, :]  # [b,1,1,n]
    pm = torch.sum(coeffs_ * sh_basis, dim=-1)

    return pm

if __name__ == "__main__":
    direction = [0., np.pi/4]
    order = 12
    coffes = get_sh_coeffs(direction, order=order)
    print(coffes.shape)
    pm = get_pm_from_sh(coffes, resolution=[640, 320], order=order)
    print(pm.shape)
    # 保存为.exr格式
    imageio.imwrite("test/sh" + "_" + str(order) + ".exr", pm.permute(1, 2, 0).numpy())
    # 保存为.png格式
    save_image(pm/pm.max(), "test/sh" + "_" + str(order) + ".png")