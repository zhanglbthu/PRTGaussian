from PIL import Image

# 打开图片
img = Image.open("assets/quantitative.png").convert("RGBA")

# 获取图片数据
data = img.getdata()

# 创建一个新的空列表，用于存储处理后的数据
new_data = []

# 遍历每个像素
for item in data:
    # 如果是透明的（即 alpha 值为 0）
    if item[3] < 1:
        # 设置为白色
        new_data.append((255, 255, 255, 255))
    else:
        # 保持原来的像素
        new_data.append(item)

# 更新图片数据
img.putdata(new_data)

# 保存处理后的图片
img.save("assets/quantitative_rgb.png")