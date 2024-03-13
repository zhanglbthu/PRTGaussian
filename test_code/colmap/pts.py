from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
    
import os
import numpy as np
    
def readColmapSceneInfo(path):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin") # images : extrinsics
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin") # cameras : intrinsics
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    translations = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        T = np.array(extr.tvec)
        translations.append(T)
    
    translations = np.array(translations)
    return translations

if __name__ == "__main__":
    path = "/home/project/gaussian-splatting/test_code/colmap/0"
    translations = readColmapSceneInfo(path)
    print(translations.shape)