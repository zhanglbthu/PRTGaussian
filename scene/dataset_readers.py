#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import OpenEXR
import Imath

from tqdm import tqdm
from my_utils.math_utils import compute_angles, compute_angle

class CameraInfo(NamedTuple):
    uid: int
    cam_id: str
    light_id: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    light_info: list = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def randamPly(path, num_pts):
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3 # * (-1.3, 1.3)
    print(f"Generated {xyz.shape[0]} points")
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    storePly(path, xyz, SH2RGB(shs) * 255)
        
    try:
        pcd = fetchPly(path)
    except:
        pcd = None
        
    return pcd

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
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

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # * scene_info definition *
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    translations = []
    light_angles = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames)):
            # 打印读取进度
            
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            
            cam_direction = c2w[:3, 3]
            
            # * 计算俯仰角和方位角
            cam_phi, cam_theta = compute_angle(cam_direction)
            
            c2w_light = np.array(frame["transform_matrix_sun"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w_light[:3, 1:3] *= -1
            
            light_direction = c2w_light[:3, 3]
            translations.append(light_direction)
            light_phi, light_theta = compute_angle(light_direction)
            
            light_angles.append([light_phi, light_theta])
            
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            
            if extension == ".png":
                image = Image.open(image_path)

                im_data = np.array(image.convert("RGBA"))

                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB") # * (H, W, 3)

                size = image.size
            
            elif extension == ".exr":
                file = OpenEXR.InputFile(image_path)
                
                dw = file.header()['dataWindow']
                size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                
                # 读取RGB颜色，不需要限制在0-1之间
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                (r, g, b) = [np.frombuffer(file.channel(Chan, pt), dtype=np.float32) for Chan in ("R", "G", "B")]
                
                r.shape = g.shape = b.shape = (size[1], size[0])
                image = np.stack([r, g, b], axis=-1) # * (H, W, 3)
            
            else:
                assert False, "Unsupported image extension: {}".format(extension)
            
            fovy = focal2fov(fov2focal(fovx, size[0]), size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            cam_phi=cam_phi, cam_theta=cam_theta, light_phi=light_phi, light_theta=light_theta,
                            image_path=image_path, image_name=image_name, extension=extension, width=size[0], height=size[1]))
            
    return cam_infos, light_angles    

def readCamerasFromOpenIlluminations(path, cam_train, cam_test, resolution_scale):
    train_cam_infos = []
    test_cam_infos = []
    
    root_folder = os.path.join(path, "Lights")
    
    # traverse lights
    for light_id in os.listdir(root_folder):
        # traverse cams
        cam_folder = os.path.join(root_folder, light_id, "raw_undistorted")
        for idx, image in enumerate(os.listdir(cam_folder)):
            
            current_step = int(light_id) * len(os.listdir(cam_folder)) + idx
            sys.stdout.write('\r')
            # 可视化进度条
            sys.stdout.write("Reading camera {}/{}".format(current_step, len(os.listdir(cam_folder)) * len(os.listdir(root_folder))))
            
            cam_id = image.split(".")[0]
        
            # read image
            img_path = os.path.join(cam_folder, image)
            
            # check if cam_id is in train or test
            if cam_id in cam_train:
                cam_info = cam_train[cam_id]
                train_cam_infos.append(initCamera(idx, cam_id, light_id, img_path, cam_info, resolution_scale))
            else:
                cam_info = cam_test[cam_id]
                test_cam_infos.append(initCamera(idx, cam_id, light_id, img_path, cam_info, resolution_scale))
            
    return train_cam_infos, test_cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", llffhold=8):
    print("Reading Training Transforms")
    cam_infos, light_angles = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    
    radius = 1
    points = []
    for phi, theta in light_angles:
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        points.append([x, y, z])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = np.array(points)
    xs, ys, zs = zip(*points)
    ax.scatter(xs, ys, zs)
    # 保存图片
    plt.savefig(os.path.join(path, "angles.png"))
    print("Saved angles.png")
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
        
    num_pts = 100_000 # change
    pcd = randamPly(ply_path, num_pts)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCameraFromJson(json_path, resolution_scale):
    with open(json_path, "r") as file:
        cam_data = json.load(file)["frames"]
        
    mask_folder = os.path.join(os.path.dirname(json_path), "obj_masks")
    
    # add obj masks
    for cam in cam_data:
        mask_path = os.path.join(mask_folder, cam.split(".")[0] + ".png")
        mask = Image.open(mask_path)
        resized_mask = mask.resize((int(mask.width / resolution_scale), int(mask.height / resolution_scale)))
        
        mask_data = np.array(resized_mask.convert("RGB")) / 255.0
        cam_data[cam]["mask"] = mask_data
    
    return cam_data
    
def initCamera(idx, cam_id, light_id, img_path, cam_info, resolution_scale):
    
    c2w = cam_info["transform_matrix"]
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])
    T = w2c[:3, 3]
    
    image_path = img_path
    image_name = Path(img_path).stem
    
    image = Image.open(image_path)
    resized_image = image.resize((int(image.width / resolution_scale), int(image.height / resolution_scale)))
    img_data = np.array(resized_image.convert("RGB"))
    
    norm_data = img_data / 255.0
    bg = np.array([0., 0., 0.])
    
    mask_data = cam_info["mask"]
    
    assert mask_data.shape[0] == img_data.shape[0] 
    
    arr = norm_data[:,:,:3] * mask_data + bg * (1 - mask_data)
    image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB") # * (H, W, 3)
    
    size = image.size
    
    fovx = cam_info["camera_angle_x"]
    fovy = focal2fov(fov2focal(fovx, size[0]), size[1])
    FovY = fovy
    FovX = fovx
    
    return CameraInfo(uid=idx, cam_id=cam_id, light_id=int(light_id), R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                      image_path=image_path, image_name=image_name, width=size[0], height=size[1])

def readOpenIlluminationInfo(source_path, num_pts, resolution_scale, eval):
    
    # init points
    ply_path = os.path.join(source_path, "points3d.ply")
    if not os.path.exists(ply_path):
        pcd = randamPly(ply_path, num_pts)
    else:
        pcd = fetchPly(ply_path)
    
    # read light pos
    light_pos_path = os.path.join(os.path.dirname(source_path), "light_pos.npy")
    light_pos = np.load(light_pos_path)
    light_info = compute_angles(light_pos)
    
    # read cam info
    cam_train_json = os.path.join(source_path, "output", "transforms_train.json")
    cam_test_json = os.path.join(source_path, "output", "transforms_test.json")
    cam_train_data = readCameraFromJson(cam_train_json, resolution_scale)
    cam_test_data = readCameraFromJson(cam_test_json, resolution_scale)

    print(f"Train: {len(cam_train_data)}, Test: {len(cam_test_data)}")
    
    cam_infos = readCamerasFromOpenIlluminations(source_path, cam_train_data, cam_test_data, resolution_scale)
    
    if eval:
        train_cam_infos = cam_infos[0]
        test_cam_infos = cam_infos[1]
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path,
                            light_info=light_info)
    
    return scene_info
    

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "OpenIllumination": readOpenIlluminationInfo
}