'''
SH implementation reference: https://github.com/abdallahdib/NextFace/blob/main/sphericalharmonics.py
'''


import math
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
import scipy.special  # 包含球谐函数相关的工具
import numpy as np


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


def pm2sh(pm, order=3, scale=10.0):
    '''
    input: pm [b,3,h,w], ensure w=2h
    output: 
        coeffs: with size of [b,3,order**2]
        pm_sh: [b,3,h,w] the env map represented by SH basis
    '''
    _, _, h, w = pm.size()
    
    theta = torch.linspace(0, math.pi, h)  # [h] from 0 to pi
    phi = torch.linspace(0, 2 * math.pi, w)  # [w] from 0 to 2pi
    theta = theta[..., None].repeat(1, w)  # [h,w]
    phi = phi[None, ...].repeat(h, 1)  # [h,w]

    dphi = 2 * math.pi / w
    dtheta = math.pi / h
    
    # change: scale the pm
    pm = pm * scale
    
    # calculate integral
    pm = pm[..., None]  # [b,3,h,w,1]

    sh_basis = []
    for l in range(order):
        for m in range(-l, l + 1):
            sh_basis.append(SH(l, m, theta, phi))
    sh_basis = torch.stack(sh_basis, dim=-1)  # [h,w,n]
    sin_theta = torch.sin(theta).unsqueeze(-1)  # [h,w,1]
    coeffs = torch.sum(pm * sh_basis * sin_theta * dtheta * dphi, dim=(2, 3))  # [b,3,n]

    # get pm represented by sh
    coeffs_ = coeffs[:, :, None, None, :]  # [b,3,1,1,n]
    pm_sh = torch.sum(coeffs_ * sh_basis, dim=-1)
    
    return coeffs, pm_sh

# 修改之前的计算SH系数的函数，使其可以处理RGB值
def compute_sh_coefficients(theta, phi, light_intensity = 1, l_max = 8):
    coefficients = np.zeros((3, (l_max + 1) ** 2), dtype=np.complex)  # 复数数组
    for l in range(0, l_max + 1):
        for m in range(-l, l + 1):
            index = l * (l + 1) + m
            Y_lm = scipy.special.sph_harm(m, l, phi, theta)
            # 计算每个通道的系数
            for channel in range(3):  # 对于R,G,B
                coefficients[channel, index] = light_intensity * Y_lm
    # 转化为tensor, shape为[b, 3, (l_max + 1) ** 2]
    coefficients = torch.from_numpy(coefficients.real).unsqueeze(0)
    
    return coefficients
            
def gen_envmap(coeffs, l_max = 8, resolution = [32, 16]):
    envmap = torch.zeros((3, resolution[1], resolution[0]))
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            theta = np.pi * j / resolution[1]
            phi = 2 * np.pi * i / resolution[0]
            
            intensity = 0.0
            for l in range(0, l_max + 1):
                for m in range(-l, l + 1):
                    index = l * (l + 1) + m
                    Y_lm = scipy.special.sph_harm(m, l, phi, theta)
                    # 计算每个通道的系数
                    for channel in range(3):
                        intensity += coeffs[0, channel, index] * Y_lm.real
            envmap[:, j, i] = intensity
    return envmap
            

if __name__ == "__main__":
    # pm_path = "grace.jpg"
    pm_path = "env_map.png"
    pm = transforms.ToTensor()(Image.open(pm_path)).unsqueeze(0)  # [1,3,h,w]
    # pm = pm
    coffes = compute_sh_coefficients(0, 0, light_intensity=0.05, l_max=8)
    pm_sh = gen_envmap(coffes, l_max=8, resolution=[128, 64])
    print(pm_sh.shape)
    save_image(pm_sh, "pm_sh.png")