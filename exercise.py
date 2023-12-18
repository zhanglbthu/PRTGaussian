import torch

if __name__ == '__main__':
    # 创建一个形状为(1, 3, 3)的张量
    x = torch.ones(1, 3, 3)
    # 创建一个形状为(10, 9)的张量
    y = torch.randn(3, 9)
    print(x)
    print(y)
    y = y.view(3, 3, 3)
    # 计算x和y的点积
    z = x * y
    # 打印z的形状
    print(z.shape)
    print(z)
    z_sum = z.sum(dim=2)
    print(z_sum.shape)
    print(z_sum)