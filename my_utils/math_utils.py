import numpy as np

def compute_angle(direction):
    # * T: translation
    # * theta: elevation angle
    # * phi: azimuth angle
    direction_normlized = direction / np.linalg.norm(direction) # (3,)
    
    phi = np.arctan2( - direction_normlized[1], direction_normlized[0]) # (-pi, pi)
    theta = np.arccos(direction_normlized[2]) # (0, pi)
    
    return phi, theta

# 有一个ndarray，shape为(N,3)，其中每一行是一个方向向量，现在编写一个函数，计算每个方向向量的俯仰角和方位角，返回一个(N,2)的ndarray
def compute_angles(directions):
    # * T: translation
    # * theta: elevation angle
    # * phi: azimuth angle
    directions_normlized = directions / np.linalg.norm(directions, axis=1, keepdims=True) # (N,3)
    
    phi = np.arctan2(directions_normlized[:,1], directions_normlized[:,0]) # (-pi, pi)
    theta = np.arccos(directions_normlized[:,2]) # (0, pi)
    
    return np.stack([phi, theta], axis=1) # (N,2)