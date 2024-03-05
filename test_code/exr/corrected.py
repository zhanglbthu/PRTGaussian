from PIL import Image
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

def PILtoTorch(pil_image):
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0 # [0, 1]
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

img_folder = '/home/project/gaussian-splatting/test_code/exr/img'
png_path = img_folder + '/origin.png'
exr_path = img_folder + '/origin.exr'

# 读取png文件
png_img = Image.open(png_path)

png_data = np.array(png_img.convert('RGBA'))

# 读取exr文件
exr_file = OpenEXR.InputFile(exr_path)
dw = exr_file.header()['dataWindow']
size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
pt = Imath.PixelType(Imath.PixelType.FLOAT)
(r, g, b) = [np.frombuffer(exr_file.channel(Chan, pt), dtype=np.float32) for Chan in ("R", "G", "B")]
r.shape = g.shape = b.shape = (size[1], size[0])
exr_data = np.stack([r, g, b], axis=-1)

print('png_data.shape:', png_data.shape)
print('exr_data.shape:', exr_data.shape)

png_torch = PILtoTorch(png_img)
print('png_torch.shape:', png_torch.shape)
torchvision.utils.save_image(png_torch, img_folder + '/origin_torch.png')

exr_torch = torch.from_numpy(exr_data).permute(2, 0, 1)

# gamma correction
exr_torch_1 = exr_torch ** (1/2.2)
exr_torch_2 = exr_torch ** (2.2)

print('exr_torch.shape:', exr_torch.shape)
torchvision.utils.save_image(exr_torch, img_folder + '/origin_torch_exr.png')
torchvision.utils.save_image(exr_torch_1, img_folder + '/origin_torch_exr_1.png')
torchvision.utils.save_image(exr_torch_2, img_folder + '/origin_torch_exr_2.png')


