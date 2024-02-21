import torch
import matplotlib.pyplot as plt

# 定义 get_grid 函数
def get_grid(resolution):
    width, height = resolution
    N = width * height
    pixels = torch.zeros((N, 2), device="cuda")
    
    for y in range(height):
        for x in range(width):
            pixels[y * width + x, 0] = x
            pixels[y * width + x, 1] = y
            
    # 将像素坐标归一化到 [0, 1] 区间
    pixels[:, 0] = pixels[:, 0] / (width - 1)
    pixels[:, 1] = pixels[:, 1] / (height - 1)    

    return pixels

# 调用函数
RESOLUTION = (100, 100)
pixels = get_grid(RESOLUTION)

# 将 pixels 从 torch 张量转换为 numpy 数组
pixels_np = pixels.cpu().numpy()

# 使用 matplotlib 绘制散点图
plt.figure(figsize=(10, 10))
plt.scatter(pixels_np[:, 0], pixels_np[:, 1], s=1)
plt.gca().invert_yaxis()  # 翻转 Y 轴，以便 (0,0) 坐标在左上角
plt.show()
# save image
plt.savefig('pixel.png')