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
import torch
# from torchvision.utils import save_image
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


from torchvision.utils import save_image
from my_utils.sh.pm2sh_v2 import get_sh_coeffs, get_pm_from_sh
import open3d as o3d
from configparser import ConfigParser
from os import makedirs
import torchvision
import json
import tinycudann as tcnn

def compute_diffuse_colors(light_coeffs, gaussians : GaussianModel, model, color_order, total_order):

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
    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, 
             debug_path=None, debug=False, extension=".png", model=None, lr=0.0001, color_order=3, total_order=9):
    # clear log.txt
    with open('log.txt', 'w') as f:
        f.write('')
        f.close()
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree) # 初始化gaussians

    scene = Scene(dataset, gaussians, extension=extension) # 初始化scene
    gaussians.training_setup(opt) # 设置优化器

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    if not debug:
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for iteration in range(first_iter, opt.iterations + 1):        

            iter_start.record()

            gaussians.update_learning_rate(iteration) # * Update learning rate

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree() # * Increase SH degree

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background
            
            light_coeffs = get_sh_coeffs(direction=(viewpoint_cam.light_phi, viewpoint_cam.light_theta), order=total_order)
            
            diffuse_colors = compute_diffuse_colors(light_coeffs, gaussians, model, color_order, total_order)
            
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, override_color=diffuse_colors) # [0, 1]
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            # 监测image是否全为0
            if torch.sum(image) == 0:
                print("image is all 0")
                print("iteration: {}".format(iteration))

            if iteration % 1000 == 0:
                sh_map_path = os.path.join(debug_path, 'sh_map')
                if not os.path.exists(sh_map_path):
                    os.makedirs(sh_map_path)
                render_path = os.path.join(debug_path, 'render')
                if not os.path.exists(render_path):
                    os.makedirs(render_path)
                # save sh_map
                sh_map = get_pm_from_sh(light_coeffs, resolution=(32, 16), order=total_order)
                save_image(sh_map, os.path.join(sh_map_path, '{0:05d}'.format(iteration) + ".png"))
                # save render image
                image_corrected = pow(image, 1.0/2.2)
                save_image(image_corrected, os.path.join(render_path, '{0:05d}'.format(iteration) + ".png"))

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image) 
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) #SRGB
            loss.backward()
            
            # 如果loss为inf或nan，停止训练
            if torch.isinf(loss) or torch.isnan(loss):
                print("loss is inf or nan, stop training")
                print("iteration: {}".format(iteration))
                break
            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background)
                                , gaussians, model, color_order, total_order)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    
                    optimizer.step()
                    optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if not debug:
        new_scene = Scene(dataset, gaussians, load_iteration=opt.iterations, shuffle=False, extension=extension)  
    else:
        new_scene = Scene(dataset, gaussians, shuffle=False)
        print("Rendering debug images")
    render_set(dataset.model_path, "train", opt.iterations, new_scene.getTrainCameras(), gaussians, pipe, background, debug_path=debug_path, model=model, color_order=color_order, total_order=total_order)
    render_set(dataset.model_path, "test", opt.iterations, new_scene.getTestCameras(), gaussians, pipe, background, debug_path=debug_path, model=model, color_order=color_order, total_order=total_order)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,
                    gaussians : GaussianModel = None, model = None, color_order=3, total_order=9):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    
                    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    # gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    light_coeffs = get_sh_coeffs(direction=(viewpoint.light_phi, viewpoint.light_theta), order=total_order)
                    diffuse_colors = compute_diffuse_colors(light_coeffs, gaussians, model, color_order, total_order)
                    image = render(viewpoint, gaussians, *renderArgs, override_color=diffuse_colors)["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def render_set(model_path, name, iteration, views, gaussians : GaussianModel, pipeline, background, debug_path=None, model = None, color_order=3, total_order=9):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    if debug_path:
        sh_map_path = os.path.join(debug_path, 'sh_map_ordered')
        if not os.path.exists(sh_map_path):
            os.makedirs(sh_map_path)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        light_coeffs = get_sh_coeffs(direction=(view.light_phi, view.light_theta), order=total_order)
        sh_map = get_pm_from_sh(light_coeffs, resolution=(32, 16), order=total_order)
        
        if debug:
            save_image(sh_map, os.path.join(sh_map_path, '{0:05d}'.format(idx) + ".png"))
        
        diffuse_colors = compute_diffuse_colors(light_coeffs, gaussians, model, color_order, total_order)
        
        rendering = render(view, gaussians, pipeline, background, override_color=diffuse_colors)["render"]
        gt = view.original_image[0:3, :, :]
        
        # correct rendering
        rendering = pow(rendering, 1.0/2.2)
        gt = pow(gt, 1.0/2.2)
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--config", type=str, required=True, default=None) # change
    
    args = parser.parse_args(sys.argv[1:])
    
    config = ConfigParser()
    config.read(args.config)
    
    root_path = config['PATH']['root_path']
    obj_name = config['NAME']['obj_name']
    out_name = config['NAME']['out_name']
    
    args.save_iterations.append(args.iterations)
    
    source_path = os.path.join(root_path, obj_name)
    out_path = os.path.join(source_path, 'out', out_name)
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    # 查看out_path下有多少个文件夹，model_path为out_path下的第几个文件夹，从0开始，比如"out_path/version_0"
    model_path = os.path.join(out_path, 'version_{}'.format(len(os.listdir(out_path))))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # 在model_path下保存config文件
    with open(os.path.join(model_path, 'config.ini'), 'w') as configfile:
        config.write(configfile)
        
    args.source_path = source_path
    args.model_path = model_path
    
    hash_path = 'config/config.json'
    with open(hash_path) as f:
        hash_config = json.load(f)
    
    # 在model_path下保存config_hash文件
    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        json.dump(hash_config, f)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    debug_path = os.path.join(model_path, 'debug')
    extension = config['SETTING']['extension']
    
    debug = config.getboolean('BOOL', 'debug')
    
    lr = config.getfloat('ARGUEMENT', 'lr')
    color_order = config.getint('ARGUEMENT', 'color_order')
    total_order = config.getint('ARGUEMENT', 'total_order')
    
    # 定义模型
    n_input_dims = 3
    dc_dims = 3 * color_order ** 2
    dm_dims = 1 * (total_order ** 2 - color_order ** 2)
    n_output_dims = dc_dims + dm_dims
    
    print("n_input_dims: {}".format(n_input_dims))
    print("n_output_dims: {}".format(n_output_dims))
    
    model = tcnn.NetworkWithInputEncoding(n_input_dims=n_input_dims, n_output_dims=n_output_dims, encoding_config=hash_config["encoding"], network_config=hash_config["network"]).to("cuda")
    
    print("debug: {}".format(debug))
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, 
             debug_path, debug, extension, model, lr, color_order, total_order)

    # All done
    print("\nTraining complete.")
