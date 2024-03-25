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
from utils.loss_utils import l1_loss, ssim, mask_loss
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
import configparser
from os import makedirs
import torchvision
import json
import tinycudann as tcnn

def compute_diffuse_colors(light, 
                           gaussians : GaussianModel, 
                           model, 
                           color_order, 
                           total_order, 
                           render_type="origin",
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

def compute_mask_colors(gaussians : GaussianModel):
    N = gaussians.get_xyz.shape[0]
    mask_colors = torch.ones((N, 3), dtype=torch.float32, device="cuda")
    
    return mask_colors

def compute_random_colors(gaussians : GaussianModel):
    N = gaussians.get_xyz.shape[0]
    random_colors = torch.rand((N, 3), dtype=torch.float32, device="cuda")
    
    return random_colors

class TrainRunner():
    def __init__(self, **kwargs):
        
        config = configparser.ConfigParser()
        config.read(kwargs['config'])
        
        self.config = config
        
        self.root_path = self.config['Data']['root_path']
        self.obj_name = self.config['Data']['obj_name']
        self.out_name = self.config['Data']['out_name']
        self.data_type = self.config['Data']['data_type']
        self.diffuse_config = self.config['DiffuseNetwork']['config_path']
        self._set_path()
        self._save_file()
        
        self.debug = False
        self.debug_from = kwargs['debug_from']
        self.test_iterations = kwargs['test_iterations']
        self.save_iterations = kwargs['save_iterations']
        self.checkpoint_iterations = kwargs['checkpoint_iterations']
        
        self.render_type = self.config['Scene']['render_type']
        self.resolution_scale = self.config.getfloat('Scene', 'resolution_scale')
        self.num_pts = self.config.getint('Scene', 'num_pts')
        self.radius = self.config.getfloat('Scene', 'radius')
        self.white_bg = self.config.getboolean('Scene', 'white_bg')
        self.light_type = self.config['Scene']['light_type']
        self.load_pts = self.config['Scene']['load_pts']
        self.opacity = self.config.getboolean('Scene', 'opacity')
        
        # 如果load_pts不为空，则optimize_pts为False
        self.optimize_pts = not bool(self.load_pts)
        print("optimize_pts: ", self.optimize_pts)
        
        # optimize
        self.batch_size = self.config.getint('Optimize', 'batch_size')
        self.lambda_mask = self.config.getfloat('Optimize', 'lambda_mask')
        
        # network
        self.lr = self.config.getfloat('DiffuseNetwork', 'lr')
        self.color_order = self.config.getint('DiffuseNetwork', 'color_order')
        self.total_order = self.config.getint('DiffuseNetwork', 'total_order')
        
        self.dataset = kwargs['dataset']
        self.opt = kwargs['opt']
        self.pipe = kwargs['pipe']
        
        self.diffuse_decoder = self._set_model(self.config, self.diffuse_config)
        
        # 打印init finished和data_type
        print("data type is: ", self.data_type)
        
    def _set_path(self):
        self.source_path = os.path.join(self.root_path, self.obj_name)
        self.out_path = os.path.join(self.source_path, 'result', self.out_name)
        
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        
        self.model_path = os.path.join(self.out_path, 'version_{}'.format(len(os.listdir(self.out_path))))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        self.debug_path = os.path.join(self.model_path, 'debug')
        if not os.path.exists(self.debug_path):
            os.makedirs(self.debug_path)
            
        self.sh_map_path = os.path.join(self.debug_path, 'sh_map')
        if not os.path.exists(self.sh_map_path):
            os.makedirs(self.sh_map_path)
        
        self.render_path = os.path.join(self.debug_path, 'render')
        if not os.path.exists(self.render_path):
            os.makedirs(self.render_path)
            
        self.ckpt_path = os.path.join(self.model_path, 'ckpt')
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
            
    def _save_file(self):
        with open(os.path.join(self.model_path, 'config.ini'), 'w') as configfile:
            self.config.write(configfile)
        
        # load diffuse config
        with open(self.diffuse_config) as f:
            diffuse_config = json.load(f)
        
        with open(os.path.join(self.model_path, 'config.json'), 'w') as f:
            json.dump(diffuse_config, f)
    
    def _set_model(self, general_config, diffuse_config):
        
        with open(diffuse_config) as f:
            diffuse_config = json.load(f)
        
        input_dim = general_config.getint('DiffuseNetwork', 'input_dim')
        color_order = general_config.getint('DiffuseNetwork', 'color_order')
        total_order = general_config.getint('DiffuseNetwork', 'total_order')
        
        dc_dim = 3 * color_order ** 2
        dm_dim = 1 * (total_order ** 2 - color_order ** 2)
        output_dim = dc_dim + dm_dim
        print("output_dim: ", output_dim)
        model = tcnn.NetworkWithInputEncoding(n_input_dims=input_dim, n_output_dims=output_dim, encoding_config=diffuse_config["encoding"], network_config=diffuse_config["network"]).to("cuda")
        return model

    def training(self):
        first_iter = 0
        tb_writer = self.prepare_output_and_logger()
        gaussians = GaussianModel(self.dataset.sh_degree)
        print("Gaussians created")
        
        scene = Scene(self.dataset, 
                      gaussians,  
                      model_path=self.model_path, 
                      source_path=self.source_path,
                      data_type=self.data_type,
                      resolution_scale=self.resolution_scale,
                      num_pts=self.num_pts,
                      radius=self.radius,
                      white_bg=self.white_bg,
                      light_type=self.light_type,
                      load_pts=self.load_pts)
        
        light_info = scene.getLightInfo()
        
        gaussians.training_setup(self.opt)
        
        bg_color = [1, 1, 1] if self.white_bg else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        
        viewpoint_stack = None
        
        if not self.debug:
            ema_loss_for_log = 0.0
            progress_bar = tqdm(range(first_iter, self.opt.iterations), desc="Training progress")
            first_iter += 1
            
            optimizer = torch.optim.Adam(self.diffuse_decoder.parameters(), lr=self.lr)
            
            for iteration in range(first_iter, self.opt.iterations + 1):        
                iter_start.record()
            
                loss = 0.0
                for batch in range(self.batch_size):
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                    
                    if (iteration - 1) == self.debug_from:
                        self.pipe.debug = True
                    
                    bg = torch.rand((3), device="cuda") if self.opt.random_background else background
                    
                    diffuse_colors = compute_diffuse_colors(light_info[viewpoint_cam.light_id] if light_info is not None else None,
                                                            gaussians, 
                                                            self.diffuse_decoder, 
                                                            self.color_order, 
                                                            self.total_order,
                                                            self.render_type,
                                                            self.data_type)
                    
                    gt_image = viewpoint_cam.original_image.to("cuda")
                    gt_mask = viewpoint_cam.mask.to("cuda")
                    
                    gt_mask = torch.cat((gt_mask, gt_mask, gt_mask), dim=0)
                    
                    render_pkg = render(viewpoint_cam, gaussians, self.pipe, bg, override_color=diffuse_colors, override_opacity=self.opacity)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    
                    random_colors = compute_random_colors(gaussians)
                    visualize = render(viewpoint_cam, gaussians, self.pipe, bg, override_color=random_colors, override_opacity=self.opacity)["render"]
                    
                    Ll1 = l1_loss(image, gt_image)
                    
                    loss += (1.0 - self.opt.lambda_dssim) * Ll1
                    
                    # if iteration % 1000 == 0 and batch == 0:
                        
                    #     gt_image += torch.abs(gt_image.min())
                    #     gt_image *= gt_mask
                    #     gt_image /= gt_image.max()
                    #     image += torch.abs(image.min())
                    #     image *= gt_mask
                    #     image /= image.max()
                        
                    #     gt_image **= 1/2.2
                    #     image **= 1/2.2
                        
                    #     save_image(torch.cat((gt_image, image), 2), os.path.join(self.render_path, '{0:05d}'.format(iteration) + ".png"))
                    #     save_image(visualize, os.path.join(self.render_path, '{0:05d}'.format(iteration) + "_visualize.png"))
                
                loss /= self.batch_size
                loss.backward()
                
                iter_end.record()
                
                with torch.no_grad():
                    ema_loss_for_log = loss.item() 
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                        progress_bar.update(10)
                    if iteration == self.opt.iterations:
                        progress_bar.close()
                    
                    self.training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), self.test_iterations, scene, (self.pipe, background), gaussians, light_info)
                    
                    if (iteration in self.save_iterations):
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)
                    
                    if iteration < self.opt.densify_until_iter and self.optimize_pts:
                        print("\n[ITER {}] Optimizing Points".format(iteration))
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                        
                        if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                            size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                            gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        
                        if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                            gaussians.reset_opacity()
                    
                    if iteration < self.opt.iterations:
                        if self.optimize_pts:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none = True)
                        
                        if self.render_type != "origin":
                            optimizer.step()
                            optimizer.zero_grad(set_to_none = True)
                    
                    if iteration in self.checkpoint_iterations:
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        gaussians_ckpt_folder = os.path.join(self.ckpt_path, "iteration_{}".format(iteration))
                        os.makedirs(gaussians_ckpt_folder, exist_ok = True)
                        torch.save((gaussians.capture(), iteration), os.path.join(gaussians_ckpt_folder, "gaussians.pth"))
                        
                        diffuse_decoder_ckpt_folder = os.path.join(self.ckpt_path, "iteration_{}".format(iteration))
                        os.makedirs(diffuse_decoder_ckpt_folder, exist_ok = True)
                        torch.save(self.diffuse_decoder.state_dict(), os.path.join(diffuse_decoder_ckpt_folder, "diffuse_decoder.pth"))

        self.render_set("train", self.opt.iterations, scene.getTrainCameras(), gaussians, self.pipe, background, light_info)
        self.render_set("test", self.opt.iterations, scene.getTestCameras(), gaussians, self.pipe, background, light_info)

    def prepare_output_and_logger(self):    
        if not self.model_path:
            if os.getenv('OAR_JOB_ID'):
                unique_str=os.getenv('OAR_JOB_ID')
            else:
                unique_str = str(uuid.uuid4())
            self.model_path = os.path.join("./output/", unique_str[0:10])
            
        # Set up output folder
        print("Output folder: {}".format(self.model_path))
        os.makedirs(self.model_path, exist_ok = True)
        with open(os.path.join(self.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(self.dataset))))

        # Create Tensorboard writer
        tb_writer = None
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(self.model_path)
        else:
            print("Tensorboard not available: not logging progress")
        return tb_writer

    def training_report(self, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderArgs,
                        gaussians : GaussianModel = None, light_info = None):
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
                        
                        diffuse_colors = compute_diffuse_colors(light_info[viewpoint.light_id] if light_info is not None else None,
                                                                gaussians, 
                                                                self.diffuse_decoder, 
                                                                self.color_order, 
                                                                self.total_order, 
                                                                self.render_type)
                        
                        image = render(viewpoint, gaussians, *renderArgs, override_color=diffuse_colors, override_opacity=self.opacity)["render"]
                        
                        random_colors = compute_random_colors(gaussians)
                        visualize = render(viewpoint, gaussians, *renderArgs, override_color=random_colors, override_opacity=self.opacity)["render"]
                        
                        gt_image = viewpoint.original_image.to("cuda")
                        gt_mask = viewpoint.mask.to("cuda")
                        gt_mask = torch.cat((gt_mask, gt_mask, gt_mask), dim=0)
                        
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        
                        if self.data_type == "NeRF":
                            image = self.correct_image(image, mask=gt_mask, scale=True, gamma=True)
                            gt_image = self.correct_image(gt_image, mask=gt_mask, scale=True, gamma=True)
                        
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            
                            # 根据iteration创建render_path下的文件夹
                            render_folder = os.path.join(self.render_path, '{}_{:d}'.format(config['name'], idx), 'renders')
                            mask_folder = os.path.join(self.render_path, '{}_{:d}'.format(config['name'], idx), 'mask')
                            if not os.path.exists(render_folder):
                                os.makedirs(render_folder)
                            if not os.path.exists(mask_folder):
                                os.makedirs(mask_folder)
                            save_image(torch.cat((gt_image, image), 2), os.path.join(render_folder, '{0:05d}'.format(iteration) + ".png"))
                            save_image(visualize, os.path.join(mask_folder, '{0:05d}'.format(iteration) + ".png"))
                            
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

    def render_set(self, name, iteration, views, gaussians : GaussianModel, pipeline, background, light_info):
        
        render_path = os.path.join(self.model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(self.model_path, name, "ours_{}".format(iteration), "gt")

        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            
            diffuse_colors = compute_diffuse_colors(light_info[view.light_id] if light_info is not None else None,
                                                    gaussians, 
                                                    self.diffuse_decoder, 
                                                    self.color_order, 
                                                    self.total_order, 
                                                    self.render_type)
            
            rendering = render(view, gaussians, pipeline, background, override_color=diffuse_colors, override_opacity=self.opacity)["render"]
            gt = view.original_image[0:3, :, :].to("cuda")
            gt_mask = view.mask.to("cuda")
            gt_mask = torch.cat((gt_mask, gt_mask, gt_mask), dim=0)
            
            rendering = self.correct_image(rendering, mask=gt_mask, scale=True, gamma=True)
            gt = self.correct_image(gt, mask=gt_mask, scale=True, gamma=True)
            
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    def correct_image(self, image, mask=None, scale=False, gamma=False, shift=False):
        if torch.min(image) < 0 and shift:
            image = image + torch.abs(torch.min(image))
        if scale:
            image = image / torch.max(image)
        if gamma:
            image = image ** (1/2.2)
        if mask is not None:
            image = image * mask
        return image
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 3_000, 5_000, 7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--config", type=str, required=True, default=None) # change
    
    args = parser.parse_args(sys.argv[1:])
    
    args.save_iterations.append(args.iterations)
    
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    trainer = TrainRunner(config = args.config,
                            debug_from = args.debug_from,
                            test_iterations = args.test_iterations,
                            save_iterations = args.save_iterations,
                            checkpoint_iterations = args.checkpoint_iterations,
                            dataset = lp.extract(args),
                            opt = op.extract(args),
                            pipe = pp.extract(args))
    
    trainer.training()

    # All done
    print("\nTraining complete.")