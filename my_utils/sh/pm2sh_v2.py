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

def get_sh_coeffs(direction=[0.,0.], order=3, scale=1.0):
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
    coeffs = sh_basis.unsqueeze(0).repeat(3, 1) * scale
    coeffs = coeffs
    
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

    # dphi = 2 * math.pi / w
    # dtheta = math.pi / h

    sh_basis = []
    for l in range(order):
        for m in range(-l, l + 1):
            sh_basis.append(SH(l, m, theta, phi))
    sh_basis = torch.stack(sh_basis, dim=-1)  # [h,w,n]

    coeffs_ = coeffs[:, None, None, :]  # [b,1,1,n]
    pm = torch.sum(coeffs_ * sh_basis, dim=-1)

    return pm

def dir2sh(direction=[0.,0.], resolution=[32, 16], order=3, scale=1.0):
    '''
    input: direction [b,3]
    output: 
        coeffs: with size of [b,3,order**2]
    '''
    phi, theta = direction
    w,h = resolution

    theta_grad = torch.linspace(0, math.pi, h)  # [h] from 0 to pi
    phi_grad = torch.linspace(0, 2 * math.pi, w)  # [w] from 0 to 2pi
    theta_grad = theta_grad[..., None].repeat(1, w)  # [h,w]
    phi_grad = phi_grad[None, ...].repeat(h, 1)  # [h,w]
    
    sh_basis = []
    for l in range(order):
        for m in range(-l, l + 1):
            sh_basis.append(SH(l, m, theta_grad, phi_grad))
    sh_basis = torch.stack(sh_basis, dim=-1)  # [h,w,n]
    
    # 根据给定的方向计算对应的像素位置
    point_x = min(max(int(w * phi / (math.pi * 2) + w / 2), 0), w-1)
    point_y = min(max(int(h * theta / math.pi), 0), h-1)
    
    # 从球谐基中提取对应位置的系数
    coeffs_at_point = sh_basis[..., point_y, point_x, :] # [n]
    
    # 将系数扩展为所需的形状 [3, n]
    coeffs = coeffs_at_point.unsqueeze(0).repeat(3, 1) * scale
    
    # get pm represented by sh
    coeffs_ = coeffs[:, None, None, :]  # [3,1,1,n]
    pm_sh = torch.sum(coeffs_ * sh_basis, dim=-1)
    
    return coeffs, pm_sh

def pm2sh(pm, order=3, direction=[np.pi/2, np.pi/2]):
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

    # dphi = 2 * math.pi / w
    # dtheta = math.pi / h
    
    # calculate integral
    pm = pm[..., None]  # [b,3,h,w,1]

    sh_basis = []
    for l in range(order):
        for m in range(-l, l + 1):
            sh_basis.append(SH(l, m, theta, phi))
    sh_basis = torch.stack(sh_basis, dim=-1)  # [h,w,n]
    sin_theta = torch.sin(theta).unsqueeze(-1)  # [h,w,1]
    # coeffs = torch.sum(pm * sh_basis * sin_theta * dtheta * dphi, dim=(2, 3))  # [b,3,n]
    
    # 根据给定的方向计算对应的像素位置
    point_x = int(w * direction[0] / (math.pi * 2) + w / 2)
    point_y = int(h * direction[1] / math.pi)

    # 从球谐基中提取对应位置的系数
    coeffs_at_point = sh_basis[..., point_y, point_x, :]

    # 将系数扩展为所需的形状 [b,3,n]
    # 假设 b 是批次大小，此处设为 1，因为我们只处理一个方向
    b = pm.shape[0]
    coeffs = coeffs_at_point.unsqueeze(0).unsqueeze(0).repeat(b, 3, 1)
    print(coeffs.shape)

    # get pm represented by sh
    coeffs_ = coeffs[:, :, None, None, :]  # [b,3,1,1,n]
    pm_sh = torch.sum(coeffs_ * sh_basis, dim=-1)
    
    return coeffs, pm_sh


if __name__ == "__main__":
    # pm_path = "grace.jpg"
    # pm = transforms.ToTensor()(Image.open(pm_path)).unsqueeze(0)  # [1,3,h,w]
    # coeffs, pm_sh_9 = pm2sh(pm, order=4)
    
    # save_image(pm_sh_9, "test/new.jpg")
    # coffes, pm = dir2sh(direction=[0, np.pi/2], resolution=[32, 16], order=3, scale=1.0)
    # print(coffes.shape)
    # save_image(pm, "test/sh_3_2048.jpg")
    coffes = get_sh_coeffs(direction=[0, np.pi/2], order=9, scale=1.0)
    print(coffes.shape)
    pm = get_pm_from_sh(coffes, resolution=[32, 16], order=9)
    print(torch.max(pm))
    print(torch.min(pm))
    save_image(pm/torch.max(pm), "test/coffes2pm_9.jpg")
    

