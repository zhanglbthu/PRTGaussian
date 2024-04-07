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
from my_utils.math_utils import compute_angles
from my_utils.sh.pm2sh_v2 import get_sh_coeffs
import torch

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
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              cam_id='', light_id=0)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def randamPly(path, num_pts, radius = 0.1):
    xyz = np.random.random((num_pts, 3)) * (2 * radius) - radius
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

def readEXRImage(image_path):
    file = OpenEXR.InputFile(image_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    (r, g, b, a) = [np.frombuffer(file.channel(Chan, pt), dtype=np.float32) for Chan in ("R", "G", "B", "A")]
    r.shape = g.shape = b.shape = a.shape = (size[1], size[0]) # 从(w, h)转换为(h, w)
    image = np.stack([r, g, b, a], axis=-1)
    return image, size

def readCamerasFromTransforms(path, transformsfile):
    cam_infos = []
    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        
        tmp_path = os.path.join(path, frames[0]["file_path"] + '.exr')
        tmp_image, _ = readEXRImage(tmp_path)
        shift_image = np.zeros((tmp_image.shape))
        
        for idx, frame in tqdm(enumerate(frames)):
            # if frame["light_idx"] > 2:
            #     continue

            image_path = os.path.join(path, frame["file_path"] + '.exr')
            image_name = Path(image_path).stem
            # if frame["light_idx"] == 0:
            #     shift_image, _ = readEXRImage(image_path)
            
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            
            image, size = readEXRImage(image_path)
            
            # image[:, :, :3] = image[:, :, :3] - 4 / 5 * shift_image[:, :, :3]
            
            fovy = focal2fov(fov2focal(fovx, size[0]), size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, cam_id="", light_id=frame["light_idx"],
                                        R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=size[0], height=size[1]))
            
    return cam_infos   

def readCamerasFromOpenIlluminations(path, cam_train, cam_test, resolution_scale, white_bg, light_type):
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
            if light_id == "001" \
                or light_id == "003" \
                or light_id == "004" \
                or light_id == "005" \
                or light_id == "006" \
                or light_id == "007" \
                or light_id == "009" \
                or light_id == "010" \
                or light_id == "012" \
                or light_id == "013":
                cam_info = cam_train[cam_id] if cam_id in cam_train else cam_test[cam_id]
                train_cam_infos.append(initCamera(idx, cam_id, light_id, img_path, cam_info, resolution_scale, white_bg))
            else:
                cam_info = cam_test[cam_id] if cam_id in cam_test else cam_train[cam_id]
                test_cam_infos.append(initCamera(idx, cam_id, light_id, img_path, cam_info, resolution_scale, white_bg))
            
    return train_cam_infos, test_cam_infos

def loadShLightCoeffs(N = 81):

    basic_coeffs = torch.zeros(1, N)

    coeffs_list = []
    for i in range(N):
        light_coeffs = basic_coeffs.clone()
        light_coeffs[0, i] += 1
        coeffs_list.append(light_coeffs.repeat(3, 1).unsqueeze(0))
    
    final_coeffs = torch.cat(coeffs_list, dim=0)
    print("final_coeffs: ", final_coeffs)
    assert final_coeffs.shape == (N, 3, N)
    return final_coeffs

def readNerfSyntheticInfo(path, num_pts, eval, radius, llffhold=8):
    print("Reading Training Transforms")
    cam_infos = readCamerasFromTransforms(path, "transforms_train.json")
    print(f"Found {len(cam_infos)} cameras")
    light_nums = cam_infos[-1].light_id + 1
    print(f"Found {light_nums} lights")
    light_pos_path = os.path.join(path, "light_pos.npy")
    
    if os.path.exists(light_pos_path):
        print("Loading light pos")
        light_pos = np.load(light_pos_path)
        light_info = []
        dirs = compute_angles(light_pos)
        for dir in dirs:
            light_info.append(get_sh_coeffs(direction=dir, order=9))
        light_info = torch.stack(light_info, dim=0)
        print(f"Loaded light info with shape {light_info.shape}")
        
    else:
        light_info = loadShLightCoeffs(N = 81)
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
        
    pcd = randamPly(ply_path, num_pts, radius=radius)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           light_info=light_info)
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
        
        mask_data = np.array(resized_mask.convert("L"))
        cam_data[cam]["mask"] = mask_data[:, :, None]
    
    return cam_data
    
def initCamera(idx, cam_id, light_id, img_path, cam_info, resolution_scale, white_bg=False):
    
    c2w = np.array(cam_info["transform_matrix"])
    w2c = np.linalg.inv(c2w)
    
    R = np.transpose(w2c[:3,:3]) 
    T = w2c[:3, 3]
    
    image_path = img_path
    image_name = Path(img_path).stem
    
    image = Image.open(image_path)
    resized_image = image.resize((int(image.width / resolution_scale), int(image.height / resolution_scale)))
    img_data = np.array(resized_image.convert("RGB"))
    
    mask_data = cam_info["mask"]
    
    if mask_data.max() == 1:
        mask_data = mask_data * 255
    
    assert mask_data.shape[0] == img_data.shape[0] 
    
    image = np.concatenate([img_data, mask_data], axis=-1)
    
    image = Image.fromarray(image, "RGBA")
    
    size = image.size
    
    fovx = cam_info["camera_angle_x"]
    fovy = focal2fov(fov2focal(fovx, size[0]), size[1])
    FovY = fovy
    FovX = fovx
    
    return CameraInfo(uid=idx, cam_id=cam_id, light_id=int(light_id)-1, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                      image_path=image_path, image_name=image_name, width=size[0], height=size[1])

def readOpenIlluminationInfo(source_path, num_pts, resolution_scale, eval, radius, white_bg, light_type):
    
    # init points
    ply_path = os.path.join(source_path, "points3d.ply")
    
    pcd = randamPly(ply_path, num_pts, radius=radius)

    # read light pos
    if light_type == "OLAT":
        light_pos_path = os.path.join(os.path.dirname(source_path), "light_pos.npy")
        light_pos = np.load(light_pos_path)
        light_info = compute_angles(light_pos)
    elif light_type == "light_pattern":
        light_pattern_path = os.path.join(os.path.dirname(source_path), "light_coeffs_9.pt")
        light_info = torch.load(light_pattern_path)
    
    # read cam info
    cam_train_json = os.path.join(source_path, "output", "transforms_train.json")
    cam_test_json = os.path.join(source_path, "output", "transforms_test.json")
    cam_train_data = readCameraFromJson(cam_train_json, resolution_scale)
    cam_test_data = readCameraFromJson(cam_test_json, resolution_scale)

    print(f"Train: {len(cam_train_data)}, Test: {len(cam_test_data)}")
    
    cam_infos = readCamerasFromOpenIlluminations(source_path, cam_train_data, cam_test_data, resolution_scale, white_bg, light_type)
    
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