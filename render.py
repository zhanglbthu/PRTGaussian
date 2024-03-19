import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from my_utils.sh.pm2sh_v2 import get_sh_coeffs
import configparser
import sys
import json
import tinycudann as tcnn
from PIL import Image
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr

def compute_diffuse_colors(light, 
                           gaussians : GaussianModel, 
                           model, 
                           color_order=9, 
                           total_order=9, 
                           render_type="not_origin",
                           data_type="NeRF"):

    if render_type == "origin":
        return None
    
    if data_type == "NeRF":
        light_coeffs = light
    elif data_type == "OpenIllumination": 
        if light.shape[0] == 3:
            light_coeffs = light.unsqueeze(0)
        else:
            light_coeffs = get_sh_coeffs(direction=light, order=total_order)
    
    xyz = gaussians.get_xyz.clone().detach()
    N = xyz.shape[0]
    
    if torch.cuda.is_available():
        xyz = xyz.to("cuda") # (N, 3)
        light_coeffs = light_coeffs.to("cuda")
        
    trans_coeffs = model(xyz)
        
    x_front = trans_coeffs[:, :3*color_order**2] . view(N, 3, color_order**2)
    x_back = trans_coeffs[:, 3*color_order**2:] . repeat(1, 3).view(N, 3, total_order**2 - color_order**2)
    d = torch.cat((x_front, x_back), dim=2)
    
    diffuse_colors = (d * light_coeffs).sum(dim=2)
    
    return diffuse_colors

def render_set(source_path, 
               name, 
               iteration, 
               views, 
               gaussians,
               diffuse_network,
               light,
               pipeline, 
               background):
    
    render_path = os.path.join(source_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(source_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        diffuse_colors = compute_diffuse_colors(light[view.light_id],
                                                gaussians,
                                                diffuse_network)
        
        rendering = render(view, gaussians, pipeline, background, override_color=diffuse_colors)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering ** (1/2.2), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt ** (1/2.2), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, 
                iteration : int, 
                pipeline : PipelineParams, 
                model_path : str,
                source_path : str,
                diffuse_network : tcnn.NetworkWithInputEncoding,
                light : torch.Tensor):
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(args=dataset, 
                      gaussians=gaussians,
                      load_iteration=iteration, 
                      shuffle=False, 
                      model_path=model_path, 
                      source_path=source_path)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(source_path, 
                   "test", 
                   scene.loaded_iter, 
                   scene.getTrainCameras(), 
                   gaussians,
                   diffuse_network,
                   light,
                   pipeline, 
                   background)

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir + "/" + fname)
        gt = Image.open(gt_dir + "/" + fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(source_path):
    test_dir = os.path.join(source_path, "test")
    full_dict = {}
    
    full_dict[source_path] = {}
    
    for method in os.listdir(test_dir):
        print("Method:", method)
        
        full_dict[source_path][method] = {}    
        
        method_dir = os.path.join(test_dir, method)
        gt_dir = os.path.join(method_dir, "gt")
        renders_dir = os.path.join(method_dir, "renders")
        renders, gts = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx]))

        print("SSIM:", sum(ssims) / len(ssims))
        print("PSNR:", sum(psnrs) / len(psnrs))
        print("LPIPS:", sum(lpipss) / len(lpipss))
        
        full_dict[source_path][method].update({"SSIM": sum(ssims) / len(ssims),
                                              "PSNR": sum(psnrs) / len(psnrs),
                                              "LPIPS": sum(lpipss) / len(lpipss)})
    
    with open(os.path.join(source_path + "/results.json"), 'w') as file:
        json.dump(full_dict[source_path], file, indent=True)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--config", type=str, required=True, default=None) # change
    
    args = parser.parse_args(sys.argv[1:])

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    config = configparser.ConfigParser()
    config.read(args.config)
    
    source_path = config["Path"]["source_path"]
    model_path = config["Path"]["model_path"]
    light_path = config["Path"]["light_path"]
    ckpt_path = os.path.join(model_path, "ckpt")
    
    input_dim = config.getint("DiffuseNetwork", "input_dim")
    output_dim = config.getint("DiffuseNetwork", "output_dim")
    config_path = config["DiffuseNetwork"]["config_path"]

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    load_iteration = searchForMaxIteration(ckpt_path)
    dn_ckpt = os.path.join(ckpt_path, "iteration_{}".format(load_iteration), "diffuse_decoder.pth")
    diffuse_network = tcnn.NetworkWithInputEncoding(n_input_dims=input_dim, 
                                                    n_output_dims=output_dim, 
                                                    encoding_config=config["encoding"],
                                                    network_config=config["network"])
    
    diffuse_network.load_state_dict(torch.load(dn_ckpt))
    light = torch.load(light_path)
    
    render_sets(model.extract(args), 
                args.iteration, 
                pipeline.extract(args), 
                model_path,
                source_path,
                diffuse_network,
                light)
    
    evaluate(source_path)