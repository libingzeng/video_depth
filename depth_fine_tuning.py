#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import cv2
import itertools
import json
import math
import os
from os.path import join as pjoin
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from typing import Dict

from utils.helpers import SuppressedStdout, dotdict, backWarp
from monodepth.depth_model_registry import get_depth_model

import optimizer
from loaders.video_dataset import VideoDataset, VideoFrameDataset, load_mask, load_raw_float32_image
from loss.joint_loss import JointLoss
from loss.loss_params import LossParams
from utils import image_io, visualization
from utils.torch_helpers import to_device
from utils import consistency as consistency
from flow import gradient
from utils.geometry import pixel_grid, pixels_to_points, to_worldspace, to_camera, project

from utils.helpers import set_requires_grad, set_requires_grad_flownet
# from third_party.flownet2.models import FlowNet2
from core.raft import RAFT
from third_party.OpticalFlowToolkit.lib.flowlib import flow_to_image

from fast_bilateral_solver import fast_bilateral_solver, fast_bilateral_solver_mask

import numpy as np
import pdb

class DepthFineTuningParams:
    """Options about finetune parameters.
    """

    @staticmethod
    def add_arguments(parser):
        parser = LossParams.add_arguments(parser)

        parser.add_argument(
            "--optimizer",
            default="Adam",
            choices=optimizer.OPTIMIZER_NAMES,
            help="optimizer to train the network",
        )
        parser.add_argument(
            "--val_epoch_freq",
            type=int,
            default=1,
            help="validation epoch frequency.",
        )
        parser.add_argument("--learning_rate", type=float, default=0,
            help="Learning rate for the training. If <= 0 it will be set"
            " automatically to the default for the specified model adapter.")
        parser.add_argument("--learning_rate_flow", type=float, default=0,
            help="Learning rate for the fine-tuning of flow net ")
        parser.add_argument("--batch_size", type=int, default=3)
        parser.add_argument("--num_epochs", type=int, default=20)

        parser.add_argument("--log_dir", help="folder to log tensorboard summary")

        parser.add_argument('--display_freq', type=int, default=100,
            help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=1,
            help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--straight_line_method', type=bool, default=False,
            help='the spatial_disparity_method or the straight_line_method')
        parser.add_argument('--pred_ref_gt_diff_weight', type=float, default=1.,
            help='the weight of the difference between the reference depth of predicted and the reference depth of ground truth calculated by straight lines')
        parser.add_argument('--pred_tgt_gt_diff_weight', type=float, default=1.,
            help='the weight of the difference between the target depth of predicted and the target depth of ground truth calculated by straight lines')
        parser.add_argument('--ref_tgt_depth_diff_weight', type=float, default=2.5,
            help='the weight of the difference between the depth of reference image and the depth of target image')
        parser.add_argument('--g1_g2_diff_weight', type=float, default=0,
            help='the weight of the difference between the two intersection points (used to finetune flow)')
        parser.add_argument('--flow_finetune_weight', type=float, default=1e-1,
            help='the weight of flow finetuning term')
        parser.add_argument('--eps', type=float, default=2e-6,
            help='epsilon used to dedicate parallel straight lines, cos_theta <= eps means two straight lines are parallel')
        parser.add_argument('--eps2', type=float, default=2e-6,
            help='epsilon used to discard small depth')
        parser.add_argument('--max_depth', type=float, default=1e3,
            help='max depth used for dedicate parallel straight lines')
        parser.add_argument('--pos_dist_min', type=float, default=0.03,
            help='minimal distance between two cameras of frame pair sampled')

        parser.add_argument('--reproj_weight', type=float, default=1.,
            help='the weight of the spatial loss')
        parser.add_argument('--disp_weight', type=float, default=1.,
            help='the weight of the disparity loss')


        parser.add_argument('--test_2_images', type=bool, default=False,
            help='just sample one pair of images for training')
        parser.add_argument('--first_image', type=int, default=0,
            help='the id of the first image')

        parser.add_argument('--flow_finetune', action='store_true')
        parser.add_argument('--is_flow_finetune', action='store_true')
        parser.add_argument('--finetune_ratio', type=int, default=3,
            help='if iter % training_ratio == 0, finetune flow, otherwise finetune depth')
        parser.add_argument('--flow_finetune_epi', action='store_true')
        parser.add_argument('--is_homography', action='store_true', \
            help='use pairs of frames after homography operation or not')
        parser.add_argument('--is_tum_scene', action='store_true', \
            help='tum scenes include ground truth of camera params and depth, so colmap process is not needed')
        parser.add_argument('--is_scannet_scene', action='store_true', \
            help='scannet scenes include ground truth of camera params and depth, so colmap process is not needed')
        parser.add_argument('--is_kitti_scene', action='store_true', \
            help='kitti eigen splits scenes include ground truth depth in .npz, so transfering it to .raw is needed')
        parser.add_argument('--test_mode', action='store_true', \
            help='simple test')
        parser.add_argument('--inference_mode', action='store_true', \
            help='given checkpoint and input images, generate depth and corresponding surface.')

        parser.add_argument('--loss_case', type=int, default=1,
            help='different loss functions')
        parser.add_argument('--parallel_lines_out', action='store_true', \
            help='mask out parallel lines, default is FALSE'
            )
        parser.add_argument('--freiburg_group', type=int, default=1,
            help='1: freiburg1, 2: freiburg2, 3: freiburg3')
        parser.add_argument('--is_pair_discard', action='store_true', \
            help='discard bad pairs or not')
        parser.add_argument("--vis_depth_scale", type=float, default=None)
        parser.add_argument("--g1_diff_threshold", type=float, default=10.)

        parser.add_argument("--frame_min", type=int, default=None)
        parser.add_argument("--frame_max", type=int, default=None)
        parser.add_argument('--depth_mean_min', type=float, default=0.0,
            help='the minimal depth average of predicted ground truth. discard the depth map with mean value less than the minimal')
        parser.add_argument('--gt_predicted_is_ready', action='store_true', \
            help='predicted ground truth depth map is ready or not')
        parser.add_argument('--gt_prediction_prepro', action='store_true', \
            help='preprocess the predicted ground truth')
        parser.add_argument('--prepro_is_ready', action='store_true', \
            help='preprocessing the predicted ground truth depth maps is ready or not')
        parser.add_argument('--preproID', type=int, default=1, \
            help='0, average; 1, median; 2, selection; 10, depth projection (median)')
        parser.add_argument('--projection_distance', type=int, default=20, \
            help='2*projection_distance neighboring frames projected to the current frame.')
        parser.add_argument('--mask_final', type=int, default=1, \
            help='0: mask_median, 1: mask_projection')
        parser.add_argument('--confidence_tolerance_median', type=float, default=0.1, \
            help='for median depth map, accumulate the number of values in the range of median_value*(1+-confidence_tolerance) as the confidence of the median value')
        parser.add_argument('--confidence_enabled', action='store_true', \
            help='use confidence map or not')
        parser.add_argument('--confidence_tolerance_projection', type=float, default=0.4, \
            help='for depth projection, accumulate the number of values in the range of median_value*(1+-confidence_tolerance) as the confidence of the median value')
        parser.add_argument('--confidence_rendered', action='store_true', \
            help='render the confidence map of the median depth along with its median map')
        parser.add_argument('--confidence_normalized', action='store_true', \
            help='normalize confidence map in [0, 1]')
        parser.add_argument('--gamma', type=float, default=1, \
            help='||log(1+gamma*G2) - log(1+gamma*D2)||, gamma can be used to adjust the sensitivity of \
                the magnitude of depth, for example, larger gamma would reduce the effect of large depth.')
        parser.add_argument('--alpha', type=float, default=1, \
            help='||log(alpha+gamma*G2) - log(alpha+gamma*D2)||')
        parser.add_argument('--upsampling_factor', type=int, default=0, \
            help='upsampling factor for fast bilateral solver')
        parser.add_argument('--border', type=int, default=0, \
            help='the number of border pixels discarded in the process of depth projection')
        parser.add_argument('--gt_prediction_grad_check', action='store_true', \
            help='discarding depth maps based on their gradients')
        parser.add_argument('--ablation_median_didabled', action='store_true', \
            help='disable median operation, just fine-tune with pairs of neigboring frames and psuedo reference candidates of each frame.')

        return parser


def log_loss_stats(
    writer: SummaryWriter,
    name_prefix: str,
    loss_meta: Dict[str, torch.Tensor],
    n: int,
    log_histogram: bool = False,
):
    """
    loss_meta: sub_loss_name: individual losses
    """
    for sub_loss_name, loss_value in loss_meta.items():
        sub_loss_full_name = name_prefix + "/" + sub_loss_name

        writer.add_scalar(
            sub_loss_full_name + "/max", loss_value.max(), n,
        )
        writer.add_scalar(
            sub_loss_full_name + "/min", loss_value.min(), n,
        )
        writer.add_scalar(
            sub_loss_full_name + "/mean", loss_value.mean(), n,
        )

        if log_histogram:
            writer.add_histogram(sub_loss_full_name, loss_value, n)


def write_summary(
    writer, mode_name, input_images, depth, metadata, n_iter
):
    DIM = -3
    B = depth.shape[0]

    inv_depth_pred = depth.unsqueeze(DIM)

    mask = torch.stack(metadata['geometry_consistency']['masks'], dim=1)

    def to_vis(x):
        return x[:8].transpose(0, 1).reshape((-1,) + x.shape[DIM:])

    writer.add_image(
        mode_name + '/image',
        vutils.make_grid(to_vis(input_images), nrow=B, normalize=True), n_iter)
    writer.add_image(
        mode_name + '/pred_full',
        vutils.make_grid(to_vis(1.0 / inv_depth_pred), nrow=B, normalize=True), n_iter)
    writer.add_image(
        mode_name + '/mask',
        vutils.make_grid(to_vis(mask.float()), nrow=B, normalize=True), n_iter)


def log_loss(
    writer: SummaryWriter,
    mode_name: str,
    loss: torch.Tensor,
    loss_meta: Dict[str, torch.Tensor],
    niters: int,
):
    main_loss_name = mode_name + "/loss"

    writer.add_scalar(main_loss_name, loss, niters)
    log_loss_stats(writer, main_loss_name, loss_meta, niters)


def make_tag(params):
    if params.test_mode:
        return ("Test_Results")
    else:
        if params.straight_line_method:
            return (
                LossParams.make_str(params)
                + f"_LR{params.learning_rate}"
                + f"_BS{params.batch_size}"
                + f"_O{params.optimizer.lower()}"
                + f"_SL{params.straight_line_method}"
                + f"_PRG{params.pred_ref_gt_diff_weight}"
                + f"_PTG{params.pred_tgt_gt_diff_weight}"
                + f"_RT{params.ref_tgt_depth_diff_weight}"
                + f"_EPS{params.eps}"
                + f"_EPS2_{params.eps2}"
                + f"_LossCase{params.loss_case}"
                + f"_TestMode{params.test_mode}"
                + f"_GDT{params.g1_diff_threshold}"
                + f"_Fmin{params.frame_min}"
                + f"_Fmax{params.frame_max}"
                + f"_DMM{params.depth_mean_min}"
                + f"_GPP{params.gt_prediction_prepro}"
                + f"_ppID{params.preproID}"
                + f"_CTM{params.confidence_tolerance_median}"
                + f"_CN{params.confidence_normalized}"
                + f"_NE{params.num_epochs}"
                + f"_Dist{params.pos_dist_min}"
                + f"_CE{params.confidence_enabled}"
                + f"_AMD{params.ablation_median_didabled}"

            )
        else:
            return (
                LossParams.make_str(params)
                + f"_LR{params.learning_rate}"
                + f"_LRF{params.learning_rate_flow}"
                + f"_BS{params.batch_size}"
                + f"_O{params.optimizer.lower()}"
                + f"_Fmin{params.frame_min}"
                + f"_Fmax{params.frame_max}"
                + f"_NE{params.num_epochs}"
            )

def compute_flow(Flownet, img1, img2_reg, H_BA, is_homography=False):
    sz = img1.size()
    
    flow = Flownet(img1, img2_reg, 12, test_mode=True)[1].permute(0, 2, 3, 1)

    if is_homography:
        x = torch.linspace(0, sz[3] - 1, sz[3]).cuda()
        y = torch.linspace(0, sz[2] - 1, sz[2]).cuda()
        fy, fx = torch.meshgrid(y, x)
        fxx = fx.clone() + flow[..., 0]
        fyy = fy.clone() + flow[..., 1]
        matmul = torch.matmul( \
                                torch.inverse(H_BA.float()), \
                                torch.cat((fxx.reshape(sz[0], 1, -1), \
                                    fyy.reshape(sz[0], 1, -1), \
                                    torch.ones(fyy.shape).reshape(sz[0], 1, -1).cuda()), \
                                    1)[:, None, :, :]
                                )
        fxxx, fyyy = matmul[:, :, 0, :] / matmul[:, :, 2, :], matmul[:, :, 1, :] / matmul[:, :, 2, :]

        flow = torch.cat(( \
                fxxx.reshape(sz[0], sz[2], sz[3], 1) - fx.reshape(1, sz[2], sz[3], 1), \
                fyyy.reshape(sz[0], sz[2], sz[3], 1) - fy.reshape(1, sz[2], sz[3], 1), \
                ), 3)

    return flow

class DepthFineTuner:
    def __init__(self, range_dir, frames, params):
        self.frames = frames
        self.params = params
        self.base_dir = params.path
        self.range_dir = range_dir
        self.out_dir = pjoin(self.range_dir, make_tag(params))
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"Fine-tuning directory: '{self.out_dir}'")

        self.checkpoints_dir = pjoin(self.out_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        model = get_depth_model(params.model_type)
        self.model = model()

        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs.")
        if num_gpus > 1:
            self.params.batch_size *= num_gpus
            print(f"Adjusting batch size to {self.params.batch_size}.")

        self.reference_disparity = {}
        self.vis_depth_scale = params.vis_depth_scale

        if self.params.flow_finetune:
            # ## flownet2
            # args = dotdict()
            # args.pretrained_model_flownet = 'checkpoints/flownet2.pth'
            # # args.pretrained_model_flownet = './results/cat/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR3e-05_LRF1e-07_BS1_Oadam_SLTrue_PRG1.0_PTG1.0_RT0.8_G121.0_FFW0.0_EPS2e-06_MD1000.0_PDM0.0_FFTrue/checkpoints/0020_flow.pth'
            # args.fp16 = False
            # args.rgb_max = 255.0
            # device = torch.device("cuda:0")
            # Flownet = FlowNet2(args)
            # print(f"Loading pretrained model from '{args.pretrained_model_flownet}'.")
            # flownet_ckpt = torch.load(args.pretrained_model_flownet)
            # Flownet.load_state_dict(flownet_ckpt["state_dict"], strict=False)
            # Flownet.to(device)

            ## raft
            is_finetuned_ckpt = 0
            args = dotdict()
            if not is_finetuned_ckpt:
                args.pretrained_model_flownet = './core/checkpoints/raft-things.pth'
            else:
                # args.pretrained_model_flownet = './results/cat/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR3e-05_LRF1e-06_BS1_Oadam_SLTrue_PRG1.0_PTG1.0_RT0.8_EPS3e-06_MD1000.0_FFTrue_EPITrue_FR1/checkpoints/0001_flow.pth'
                args.pretrained_model_flownet = './results/cat/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR3e-05_LRF1e-06_BS1_Oadam_SLTrue_PRG1.0_PTG1.0_RT0.8_EPS3e-06_MD1000.0_FFTrue_EPITrue_GG1.0_FR1_HMFalse_iter120/checkpoints/00010_flow.pth'
            args.small = False
            args.epsilon = 1e-8
            args.dropout = 0.0
            args.mixed_precision = False
            device = torch.device("cuda")
            Flownet = torch.nn.DataParallel(RAFT(args)) ## important!!! DataParallel
            print(f"Loading pretrained model from '{args.pretrained_model_flownet}'.")
            flownet_ckpt = torch.load(args.pretrained_model_flownet)
            if not is_finetuned_ckpt:
                Flownet.load_state_dict(flownet_ckpt)
            else:
                Flownet.load_state_dict(flownet_ckpt['state_dict']) # for finetuned ckpt
            Flownet.to(device)

            self.flow = Flownet
            self.flow.train()
            set_requires_grad(self.flow, True)
            self.is_visualize = True


    def save_depth(self, dir: str = None, frames=None):
        if dir is None:
            dir = self.out_dir
        if frames is None:
            frames = self.frames

        color_fmt = pjoin(self.base_dir, "color_down", "frame_{:06d}.raw")
        depth_dir = pjoin(dir, "depth")
        depth_fmt = pjoin(depth_dir, "frame_{:06d}")

        dataset = VideoFrameDataset(color_fmt, frames)
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=4
        )

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.model.eval()

        os.makedirs(depth_dir, exist_ok=True)
        for data in data_loader:
            data = to_device(data)
            stacked_images, metadata = data
            frame_id = metadata["frame_id"][0]

            depth = self.model.forward(stacked_images, metadata)

            depth = depth.detach().cpu().numpy().squeeze()
            inv_depth = 1.0 / depth

            image_io.save_raw_float32_image(
                depth_fmt.format(frame_id) + ".raw", inv_depth)

            # inv_depth_vis = visualization.visualize_depth(
            #     inv_depth, depth_min=0, depth_max=self.vis_depth_scale
            # )
            # cv2.imwrite(depth_fmt.format(frame_id) + ".png", inv_depth_vis)

        with SuppressedStdout():
            visualization.visualize_depth_dir(depth_dir, depth_dir, force=True)


    def inference_depth(self, dir, frames=None):
        # Choose algorithm and set the checkpoint in the path
        # /Volumes/Elements/temp/view_synthesis/consistent_depth0/monodepth/mannequin_challenge_model.py (for MC)
        # is_test = True
        # is_cvd = True means cvd depth estimation, otherwise our depth estimation
        # scene_name, and experiment_name are used to select checkpoint
        
        if frames is None:
            frames = self.frames
        meta_file = pjoin(self.range_dir, "metadata_scaled.npz")

        inference_depth_raw_dir = os.path.join(dir, 'depth_raw')
        inference_depth_vis_dir = os.path.join(dir, 'depth_vis')
        inference_depth_vis_rgb_dir = os.path.join(dir, 'depth_vis_rgb')
        inference_depth_vis_rgb_alpha_dir = os.path.join(dir, 'depth_vis_rgb_alpha')
        os.makedirs(inference_depth_raw_dir, exist_ok=True)
        os.makedirs(inference_depth_vis_dir, exist_ok=True)
        os.makedirs(inference_depth_vis_rgb_dir, exist_ok=True)
        os.makedirs(inference_depth_vis_rgb_alpha_dir, exist_ok=True)

        color_fmt = pjoin(self.base_dir, "color_full", "frame_{:06d}.png")
        depth_raw_fmt = pjoin(inference_depth_raw_dir, "frame_{:06d}.raw")
        depth_vis_fmt = pjoin(inference_depth_vis_dir, "frame_{:06d}.png")
        depth_vis_rgb_fmt = pjoin(inference_depth_vis_rgb_dir, "frame_{:06d}_rgb.png")
        depth_vis_rgb_alpha_fmt = pjoin(inference_depth_vis_rgb_alpha_dir, "frame_{:06d}_rgb_alpha.png")

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # just used to get intrinsics [fx, fy, cx, cy] from training data
        # and then process it to test version
        predicted_suffix = "_eps2_{}_gdt_{}_dmm_{}".format(self.params.eps2, self.params.g1_diff_threshold, self.params.depth_mean_min)
        dataset = VideoDataset(self.base_dir, meta_file, params=self.params, suffix=predicted_suffix)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        size_old = cv2.imread(pjoin(self.base_dir, "color_down_png", "frame_{:06d}.png".format(0)), cv2.IMREAD_UNCHANGED).shape[:2][::-1]
        size_new = cv2.imread(pjoin(self.base_dir, "color_full", "frame_{:06d}.png".format(0)), cv2.IMREAD_UNCHANGED).shape[:2][::-1]
        intrinsics = []
        for data in data_loader:
            data = to_device(data)
            stacked_images, metadata = data
            intrinsics_raw = data[1]['intrinsics']
            for i in range(intrinsics_raw.shape[1]):
                ratio = to_device(torch.from_numpy(np.array(size_new) / np.array((size_old[0], size_old[1]))).unsqueeze(1))
                # pdb.set_trace()
                fxy = intrinsics_raw[0][i][:2] * ratio
                cxy = intrinsics_raw[0][i][2:] * ratio
                intrinsics.append(torch.cat((fxy, cxy), 1).unsqueeze(0))
            break


        dataset = VideoFrameDataset(color_fmt, frames)
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=4
        )
        self.model.eval()
        point_cloud = []
        for data in data_loader:
            data = to_device(data)
            stacked_images, metadata = data
            frame_id = metadata["frame_id"][0]

            depth_ = self.model.forward(torch.nn.functional.interpolate(stacked_images, (512, 512)), metadata)
            depth = torch.nn.functional.interpolate(depth_.unsqueeze(0).unsqueeze(0), tuple((size_new[1], size_new[0])))
            
            B, N, H, W = depth.shape
            pixels = pixel_grid(1, (H, W))
            # pdb.set_trace()
            points_cam_t = pixels_to_points(intrinsics[0][:, 0, :], depth, pixels)
            points_cam = points_cam_t.squeeze()
            # pdb.set_trace()
            if frame_id < 103:
                alpha_dir = '../../shape_material/nerfactor/output/surf/vasedeck/temp_from_logan/train_{}/alpha.png'.format(str(frame_id.cpu().numpy()).zfill(3))
                alpha = cv2.imread(alpha_dir)
                points_cam_masked = alpha * points_cam.permute(1, 2, 0).detach().cpu().numpy()
                points_cam_masked_norm = (points_cam_masked - points_cam_masked.min()) / (points_cam_masked.max() - points_cam_masked.min())
                cv2.imwrite(depth_vis_rgb_alpha_fmt.format(frame_id), (points_cam_masked_norm* 255)[:, :, (2, 1, 0)]) # [:, :, (2, 1, 0)] RGB2BGR
            points_cam_norm = (points_cam - points_cam.min()) / (points_cam.max() - points_cam.min())
            surf = points_cam_norm.permute(1, 2, 0).detach().cpu().numpy().squeeze()
            cv2.imwrite(depth_vis_rgb_fmt.format(frame_id), (surf* 255)[:, :, (2, 1, 0)])

            depth = depth.detach().cpu().numpy().squeeze()
            inv_depth = 1.0 / depth
            image_io.save_raw_float32_image(
                depth_raw_fmt.format(frame_id), inv_depth)

            metadata_dir = '../../shape_material/nerfactor/downloads/real-images/vasedeck/train_{}/metadata.json'.format(str(frame_id.cpu().numpy()).zfill(3))
            with open(metadata_dir, "r") as f:
                metadata = json.load(f)
            cam_to_world = np.array([float(x) for x in metadata['cam_transform_mat'].split(',')]).reshape(4, 4)
            points_world = to_worldspace(points_cam_t, torch.tensor(cam_to_world[None, :3, :4]).cuda())[0]
            point_cloud.append(points_world.squeeze().detach().cpu().numpy())
            # pdb.set_trace()
            # point_cloud.append(points_cam_t.squeeze().detach().cpu().numpy())
            # point_cloud.append(points_cam_norm.squeeze().detach().cpu().numpy())
        
        point_cloud_arr = np.asarray(point_cloud)
        point_cloud_dir = '../../shape_material/nerfactor/output/surf/vasedeck/vasedeck_point_cloud.npz'
        np.savez(point_cloud_dir, points=point_cloud_arr)

        # Count the total number of points based on alpha files
        surf_folder_dir = '../../shape_material/nerfactor/output/surf/vasedeck/temp_from_logan'
        surf_file_list = sorted(os.listdir(surf_folder_dir))
        
        PointsNum = 0
        for sf in surf_file_list:
            if 'asc' in sf:
                continue
            alpha_file_dir = os.path.join(surf_folder_dir, sf, 'alpha.png')
            alpha = cv2.imread(alpha_file_dir)
            NUM = (alpha[:, :, 0] != 0).sum()
            PointsNum += NUM

        pdb.set_trace()
        pts_file_dir = '../../shape_material/nerfactor/output/surf/vasedeck/vasedeck_point_cloud_alpha.asc'
        os.system('rm -rf {}'.format(pts_file_dir))
        pts_file = open(pts_file_dir, 'a')
        pts_file.write('{}\n'.format(PointsNum))
        # for f in range(point_cloud_arr.shape[0]):
        f = 0
        for data in data_loader:
            stacked_images, metadata = data
            # pdb.set_trace()
            rgb = stacked_images.detach().cpu().numpy()
            frame_id = metadata["frame_id"][0]
            if frame_id < 103:
                alpha_dir = '../../shape_material/nerfactor/output/surf/vasedeck/temp_from_logan/train_{}/alpha.png'.format(str(frame_id.cpu().numpy()).zfill(3))
                alpha = cv2.imread(alpha_dir)
                for w in range(point_cloud_arr.shape[2]):
                    for h in range(point_cloud_arr.shape[3]):
                        # pdb.set_trace()
                        if alpha[w, h, 0] != 0:
                            data_str = '{} {} {} 255 {} {} {}\n'.format(point_cloud_arr[f, 0, w, h], point_cloud_arr[f, 1, w, h], point_cloud_arr[f, 2, w, h], \
                                int(rgb[0, 0, w, h] * 255), int(rgb[0, 1, w, h] * 255), int(rgb[0, 2, w, h] * 255))
                            pts_file.write(data_str)
                f = f + 1
                if f > 1:
                    break
        pts_file.close()

        with SuppressedStdout():
            visualization.visualize_depth_dir(inference_depth_raw_dir, inference_depth_vis_dir, force=True)

    def fine_tune(self, writer=None):
        meta_file = pjoin(self.range_dir, "metadata_scaled.npz")

        if self.params.test_2_images:
            training_shuffle = False
        else:
            training_shuffle = True

        predicted_suffix = "_eps2_{}_gdt_{}_dmm_{}".format(self.params.eps2, self.params.g1_diff_threshold, self.params.depth_mean_min)

        dataset = VideoDataset(self.base_dir, meta_file, params=self.params, suffix=predicted_suffix)
        
        # pdb.set_trace()

        train_data_loader = DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            shuffle=training_shuffle,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )
        val_data_loader = DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        criterion = JointLoss(self.params,
            parameters_init=[p.clone() for p in self.model.parameters()])
        criterion_l1  = torch.nn.L1Loss()

        if writer is None:
            log_dir = pjoin(self.out_dir, "tensorboard")
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)

        opt = optimizer.create(
            self.params.optimizer,
            self.model.parameters(),
            self.params.learning_rate,
            betas=(0.9, 0.999)
        )

        if self.params.straight_line_method:
            if self.params.flow_finetune:
                opt_flow = optimizer.create(
                    self.params.optimizer,
                    self.flow.parameters(),
                    self.params.learning_rate_flow,
                    betas=(0.9, 0.999)
                )
                self.flow.train()

        eval_dir = pjoin(self.out_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)

        self.model.train()

        def suffix(epoch, niters):
            return "_e{:04d}_iter{:06d}".format(epoch, niters)

        def validate(epoch, niters):
            loss_meta = self.eval_and_save(
                criterion, val_data_loader, suffix(epoch, niters)
            )
            if writer is not None:
                log_loss_stats(
                    writer, "validation", loss_meta, epoch, log_histogram=True
                )
            print(f"Done Validation for epoch {epoch} ({niters} iterations)")

        # self.vis_depth_scale = None
        validate(0, 0)

        # Training loop.
        total_iters = 1
        
        finetuning_start_time = time.perf_counter()

        for epoch in range(self.params.num_epochs):
            epoch_start_time = time.perf_counter()
            for data in train_data_loader:
                data = to_device(data)
                stacked_img, metadata = data

                if self.params.test_2_images:
                    index1 = data[1]['geometry_consistency']['indices'][0][0]
                    index2 = data[1]['geometry_consistency']['indices'][0][1]
                    # if abs(index1 - index2) < 10:
                    #     continue
                    # if ~(index1 == 40 and index2 == 44):
                    if ~(index1 == 321 and index2 == 323):
                    # if ~(index1 == 276 and index2 == 280):
                    # if ~(index1 < 260 and index2 < 260):
                        continue

                if self.params.straight_line_method:
                    if self.params.flow_finetune:
                        if total_iters == 1:
                            # pdb.set_trace()
                            dim = (stacked_img.shape[3], stacked_img.shape[4])
                            self.backWarp = backWarp(dim, torch.device('cuda'))

                        if total_iters % self.params.finetune_ratio == 0:
                            opt_flow.zero_grad()
                            self.flow.train()
                            set_requires_grad(self.flow, True)
                            # set_requires_grad_flownet(self.flow, True)
                            self.params.is_flow_finetune = True
                            self.model.eval()
                            set_requires_grad(self.model, False)
                        else:
                            opt.zero_grad()
                            self.model.train()
                            set_requires_grad(self.model, True)
                            self.params.is_flow_finetune = False
                            self.flow.eval()
                            set_requires_grad(self.flow, False)
                        
                        hba0 = metadata['geometry_consistency']['hbas'][0]['hba']
                        hba1 = metadata['geometry_consistency']['hbas'][1]['hba']
                        reg0 = metadata['geometry_consistency']['regs'][:, 0, :, :] * 255.
                        reg1 = metadata['geometry_consistency']['regs'][:, 1, :, :] * 255.
                        img1 = torch.cat([stacked_img[:, 0, ...], stacked_img[:, 1, ...]], 0) * 255.
                        img2_reg = torch.cat([reg0, reg1], 0)
                        if not self.params.is_homography:
                            img2 = torch.cat([stacked_img[:, 1, ...], stacked_img[:, 0, ...]], 0) * 255.
                            img2_reg = img2
                        hba = torch.cat([torch.unsqueeze(hba0, 1), torch.unsqueeze(hba1, 1)], 0)
                        flow = compute_flow(self.flow, img1, img2_reg, hba, self.params.is_homography)
                        if not self.params.is_homography:
                            img1_warped = self.backWarp(img2_reg, flow.permute(0, 3, 1, 2))
                        else:
                            img1_warped = self.backWarp(img1[[1, 0], ...], flow.permute(0, 3, 1, 2))

                        colors = [stacked_img[:, 0, ...].permute(0, 2, 3, 1)[0], stacked_img[:, 1, ...].permute(0, 2, 3, 1)[0]]
                        flows = [flow[0], flow[1]]
                        masks = consistency.consistent_flow_masks(flows, colors, 1, 1)
                        # pdb.set_trace()
                        flows = [flow[0][None, ...].permute(0, 3, 1, 2), flow[1][None, ...].permute(0, 3, 1, 2)]
                        masks = [masks[0][:, None, ...].float(), masks[1][:, None, ...].float()]

                        index0 = metadata['geometry_consistency']['indices'][0][0]
                        index1 = metadata['geometry_consistency']['indices'][0][1]
                        # if self.is_visualize:
                        if self.is_visualize and (index0 == 76 and index1 == 78):
                            vis_dir = pjoin(self.out_dir, "vis")
                            os.makedirs(vis_dir, exist_ok=True)

                            img1_dir = pjoin(vis_dir, "epoch_{}_total_iters_{}_img_{:06d}.png".format(epoch, total_iters, index0))
                            img2_dir = pjoin(vis_dir, "epoch_{}_total_iters_{}_img_{:06d}.png".format(epoch, total_iters, index1))
                            img2_reg_dir = pjoin(vis_dir, "epoch_{}_total_iters_{}_img_{:06d}_reg.png".format(epoch, total_iters, index1))
                            img1_reg_dir = pjoin(vis_dir, "epoch_{}_total_iters_{}_img_{:06d}_reg.png".format(epoch, total_iters, index0))
                            flow_dir = pjoin(vis_dir, "epoch_{}_total_iters_{}_flow_{:06d}_{:06d}.png".format(epoch, total_iters, index0, index1))
                            flow_dir_pre = pjoin(vis_dir, "epoch_{}_total_iters_{}_flow_{:06d}_{:06d}_pre.png".format(epoch, total_iters, index0, index1))
                            img1_warped_dir = pjoin(vis_dir, "epoch_{}_total_iters_{}_img_{:06d}_warped.png".format(epoch, total_iters, index0))
                            img2_warped_dir = pjoin(vis_dir, "epoch_{}_total_iters_{}_img_{:06d}_warped.png".format(epoch, total_iters, index1))
                            mask_dir = pjoin(vis_dir, "epoch_{}_total_iters_{}_mask_{:06d}_{:06d}.png".format(epoch, total_iters, index0, index1))
                            mask_dir_pre = pjoin(vis_dir, "epoch_{}_total_iters_{}_mask_{:06d}_{:06d}_pre.png".format(epoch, total_iters, index0, index1))

                            # pdb.set_trace()
                            vis = flow_to_image(flow.data.cpu().numpy()[0])
                            vis_pre = flow_to_image(metadata['geometry_consistency']['flows'][0][0].permute(1, 2, 0).data.cpu().numpy())
                            cv2.imwrite(img1_dir, cv2.cvtColor((img1[0]*masks[0][0]).permute(1, 2, 0).data.cpu().numpy(), cv2.COLOR_RGB2BGR))
                            cv2.imwrite(img2_dir, cv2.cvtColor((img1[1]*masks[1][0]).permute(1, 2, 0).data.cpu().numpy(), cv2.COLOR_RGB2BGR))
                            cv2.imwrite(img2_reg_dir, cv2.cvtColor(img2_reg[0].permute(1, 2, 0).data.cpu().numpy(), cv2.COLOR_RGB2BGR))
                            cv2.imwrite(img1_reg_dir, cv2.cvtColor(img2_reg[1].permute(1, 2, 0).data.cpu().numpy(), cv2.COLOR_RGB2BGR))
                            cv2.imwrite(flow_dir, vis)
                            cv2.imwrite(flow_dir_pre, vis_pre)
                            cv2.imwrite(img1_warped_dir, cv2.cvtColor((img1_warped[0]*masks[0][0]).permute(1, 2, 0).data.cpu().numpy(), cv2.COLOR_RGB2BGR))
                            cv2.imwrite(img2_warped_dir, cv2.cvtColor((img1_warped[1]*masks[1][0]).permute(1, 2, 0).data.cpu().numpy(), cv2.COLOR_RGB2BGR))
                            cv2.imwrite(mask_dir, masks[0][0].permute(1, 2, 0).data.cpu().numpy() * 255.)
                            cv2.imwrite(mask_dir_pre, metadata['geometry_consistency']['masks'][0][0].permute(1, 2, 0).data.cpu().numpy() * 255.)

                        # warp_loss = (criterion_l1(img1[0]*masks[0][0], img1_warped[0]*masks[0][0]) + \
                        #             criterion_l1(img1[1]*masks[1][0], img1_warped[1]*masks[1][0])) / 2.
                        metadata['geometry_consistency']['flows'] = flows
                        metadata['geometry_consistency']['masks'] = masks
                    else:
                        opt.zero_grad()
                        self.model.train()
                        set_requires_grad(self.model, True)
                else:
                    opt.zero_grad()
                    self.model.train()
                    set_requires_grad(self.model, True)

                depth = self.model(stacked_img, metadata)
                loss, loss_meta = criterion(
                    depth, metadata, parameters=self.model.parameters())
                # if self.params.flow_finetune:
                #     loss += warp_loss
                #     loss_meta['warp'] = warp_loss

                pairs = metadata['geometry_consistency']['indices']
                pairs = pairs.cpu().numpy().tolist()

                print(f"Epoch = {epoch}, pairs = {pairs}, loss = {loss[0]}")
                if torch.isnan(loss):
                    del loss
                    print("Loss is NaN. Skipping.")
                    continue

                loss.backward()

                if self.params.straight_line_method:
                    if self.params.flow_finetune:
                        if total_iters % self.params.finetune_ratio == 0:
                            opt_flow.step()
                        else:
                            opt.step()
                    else:
                        opt.step()
                else:
                    opt.step()

                total_iters += stacked_img.shape[0]

                if writer is not None and total_iters % self.params.print_freq == 0:
                    log_loss(writer, 'Train', loss, loss_meta, total_iters)
                
                if writer is not None and total_iters % self.params.display_freq == 0:
                    write_summary(
                        writer, 'Train', stacked_img, depth, metadata, total_iters
                    )

                if self.params.test_2_images:
                    if (index1 == 321 and index2 == 323):
                    # if (index1 == 276 and index2 == 280):
                #     # if (index1 == 40 and index2 == 44):
                #     # if (index1 == self.params.first_image) and abs(index1 - index2) >= 4:
                        break

            epoch_end_time = time.perf_counter()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch} took {epoch_duration:.2f}s.")

            if (epoch + 1) % self.params.val_epoch_freq == 0:
                validate(epoch + 1, total_iters)

            if (epoch + 1) % self.params.save_epoch_freq == 0:
                file_name = pjoin(self.checkpoints_dir, f"{epoch + 1:04d}.pth")
                self.model.save(file_name)

                if self.params.straight_line_method:
                    if self.params.flow_finetune:
                        # save flow model
                        file_flow_name = pjoin(self.checkpoints_dir, f"{epoch + 1:04d}_flow.pth")
                        # pdb.set_trace()
                        state_dict = {'state_dict': self.flow.state_dict()}
                        torch.save(state_dict, file_flow_name)
                    else:
                        pass

        finetuning_end_time = time.perf_counter()
        finetuning_duration = finetuning_end_time - finetuning_start_time
        print(f"timing--Finetuning took {finetuning_duration:.2f}s.")

        # Validate the last epoch, unless it was just done in the loop above.
        if self.params.num_epochs % self.params.val_epoch_freq != 0:
            validate(self.params.num_epochs, total_iters)

        print("Finished Training")

    def eval_and_save(self, criterion, data_loader, suf) -> Dict[str, torch.Tensor]:
        """
        Note this function asssumes the structure of the data produced by data_loader
        """
        N = len(data_loader.dataset)
        loss_dict = {}
        saved_frames = set()
        total_index = 0
        max_frame_index = 0
        all_pairs = []

        eval_save_start_time = time.perf_counter()

        for _, data in zip(range(N), data_loader):
            # pdb.set_trace()
            data = to_device(data)
            stacked_img, metadata = data

            with torch.no_grad():
                depth = self.model(stacked_img, metadata)

            batch_indices = (
                metadata["geometry_consistency"]["indices"].cpu().numpy().tolist()
            )

            # Update the maximum frame index and pairs list.
            max_frame_index = max(max_frame_index, max(itertools.chain(*batch_indices)))
            all_pairs += batch_indices

            # Compute and store losses.
            _, loss_meta = criterion(
                depth, metadata, parameters=self.model.parameters(),
            )

            for loss_name, losses in loss_meta.items():
                if loss_name not in loss_dict:
                    loss_dict[loss_name] = {}

                for indices, loss in zip(batch_indices, losses):
                    loss_dict[loss_name][str(indices)] = loss.item()

            # Save depth maps.
            inv_depths_batch = 1.0 / depth.cpu().detach().numpy()
            if self.vis_depth_scale is None:
                # Single scale for the whole dataset.
                self.vis_depth_scale = inv_depths_batch.max()
                # pdb.set_trace()

            for inv_depths, indices in zip(inv_depths_batch, batch_indices):
                for inv_depth, index in zip(inv_depths, indices):
                    # Only save frames not saved before.
                    if index in saved_frames:
                        continue
                    saved_frames.add(index)

                    fn_pre = pjoin(
                        self.out_dir, "eval", "depth_{:06d}{}".format(index, suf)
                    )
                    image_io.save_raw_float32_image(fn_pre + ".raw", inv_depth)

                    # inv_depth_vis = visualization.visualize_depth(
                    #     inv_depth, depth_min=0, depth_max=self.vis_depth_scale
                    # )
                    inv_depth_vis = visualization.visualize_depth(inv_depth)
                    cv2.imwrite(fn_pre + ".png", inv_depth_vis)
                total_index += 1

        loss_meta = {
            loss_name: torch.tensor(tuple(loss.values()))
            for loss_name, loss in loss_dict.items()
        }
        loss_dict["mean"] = {k: v.mean().item() for k, v in loss_meta.items()}

        with open(pjoin(self.out_dir, "eval", "loss{}.json".format(suf)), "w") as f:
            json.dump(loss_dict, f)

        # Print verbose summary to stdout.
        index_width = int(math.ceil(math.log10(max_frame_index)))
        loss_names = list(loss_dict.keys())
        loss_names.remove("mean")
        loss_format = {}
        for name in loss_names:
            max_value = max(loss_dict[name].values())
            if np.isnan(max_value):
                width = float('NaN')
            else:
                width = math.ceil(math.log10(max_value))
            loss_format[name] = f"{width+7}.6f"

        for pair in sorted(all_pairs):
            line = f"({pair[0]:{index_width}d}, {pair[1]:{index_width}d}): "
            line += ", ".join(
                [f"{name}: {loss_dict[name][str(pair)]:{loss_format[name]}}"
                for name in loss_names]
            )
            print(line)

        print("Mean: " + " " * (2 * index_width) + ", ".join(
            [f"{name}: {loss_dict[name][str(pair)]:{loss_format[name]}}"
            for name in loss_names]
        ))

        eval_save_end_time = time.perf_counter()
        eval_save_duration = eval_save_end_time - eval_save_start_time
        print(f"timing---eval_save took {eval_save_duration:.2f}s.")


        return loss_meta


    def test(self, writer=None):
        meta_file = pjoin(self.range_dir, "metadata_scaled.npz")
        # meta_file = pjoin(self.range_dir, "metadata.npz")

        predicted_suffix = "_eps2_{}_gdt_{}_dmm_{}".format(self.params.eps2, self.params.g1_diff_threshold, self.params.depth_mean_min)

        dataset = VideoDataset(self.base_dir, meta_file, params=self.params, suffix=predicted_suffix)
        test_data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        criterion = JointLoss(self.params,
            parameters_init=[p.clone() for p in self.model.parameters()])
        criterion_l1  = torch.nn.L1Loss()

        test_dir = pjoin(self.out_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        eval_dir = pjoin(self.out_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        depth_gt_predicted_dir = pjoin(self.base_dir, 'depth_gt_predicted{}'.format(predicted_suffix))
        os.makedirs(depth_gt_predicted_dir, exist_ok=True)
        depth_gt_predicted_png_dir = pjoin(self.base_dir, 'depth_gt_predicted_png{}'.format(predicted_suffix))
        os.makedirs(depth_gt_predicted_png_dir, exist_ok=True)
        mask_rectified = pjoin(self.base_dir, 'mask_rectified{}'.format(predicted_suffix))
        os.makedirs(mask_rectified, exist_ok=True)
        points_cloud_dir = pjoin(self.base_dir, 'points_cloud')
        os.makedirs(points_cloud_dir, exist_ok=True)

        self.model.eval()
        # self.model.train()

        def suffix(epoch, niters):
            return "_e{:04d}_iter{:06d}".format(epoch, niters)

        def validate(epoch, niters):
            loss_meta = self.eval_and_save(
                criterion, test_data_loader, suffix(epoch, niters)
            )
            if writer is not None:
                log_loss_stats(
                    writer, "validation", loss_meta, epoch, log_histogram=True
                )
            print(f"Done Validation for epoch {epoch} ({niters} iterations)")

        # self.vis_depth_scale = 1.5639507 # None
        # validate(0, 0)
        # pdb.set_trace()
        # depth_g1_dir = pjoin(self.base_dir, 'depth_g1')
        # depth_g1_fn = pjoin(self.base_dir, 'depth_g1_list.json')
        # if True:#not os.path.isfile(depth_g1_fn):
        #     depth_g1_list = os.listdir(depth_g1_dir)
        #     depth_g1_indices = []
        #     for l in depth_g1_list:
        #         index1 = int(l.split('.')[0].split('_')[3])
        #         index2 = int(l.split('.')[0].split('_')[4])
        #         depth_g1_indices.append([index1, index2])
        #     import json
        #     fw = open(depth_g1_fn, 'w')
        #     json.dump(depth_g1_indices, fw)
        # pdb.set_trace()

        epoch_start_time = time.perf_counter()
        for data in test_data_loader:
            
            index1 = data[1]['geometry_consistency']['indices'][0][0]
            index2 = data[1]['geometry_consistency']['indices'][0][1]
            if self.params.test_2_images:
                # if abs(index1 - index2) < 10:
                #     continue
                # if ~(index1 == 16 and index2 == 20):
                # if ~(index1 == 40 and index2 == 44):
                if ~(index1 == 170 and index2 == 171):
                # if ~(index1 == 200 and index2 == 204):
                # if ~(index1 == 228 and index2 == 232):
                # if ~(index1 == 276 and index2 == 280):
                # if ~(index1 == 296 and index2 == 300):
                # if ~(index1 == 397 and index2 == 399):
                # if ~(index1 == 276 and index2 == 284):
                # if ~(index1 == 364 and index2 == 365):
                # if ~(index1 == self.params.first_image) or abs(index1 - index2) < 4:
                    continue

            data = to_device(data)
            stacked_img, metadata = data
            with torch.no_grad():
                depth = self.model(stacked_img, metadata)

            # pdb.set_trace()
            ## visualize depth 
            inv_depth = 1.0 / depth.cpu().detach().numpy()
            inv_depth0 = inv_depth[0, 0, ...]# * data[1]['geometry_consistency']['masks'][0].cpu().numpy()[0, 0, ...]
            inv_depth1 = inv_depth[0, 1, ...]# * data[1]['geometry_consistency']['masks'][1].cpu().numpy()[0, 0, ...]            
            if self.vis_depth_scale is None:
                self.vis_depth_scale = 1.5639507 # max(inv_depth0.max(), inv_depth1.max())
            inv_depth_vis = visualization.visualize_depth(inv_depth0, depth_min=0, depth_max=self.vis_depth_scale)
            cv2.imwrite(pjoin(test_dir, "test_depth_frame_{}.png".format(index1)), inv_depth_vis)
            inv_depth_vis = visualization.visualize_depth(inv_depth1, depth_min=0, depth_max=self.vis_depth_scale)
            cv2.imwrite(pjoin(test_dir, "test_depth_frame_{}.png".format(index2)), inv_depth_vis)

            loss, loss_meta = criterion(
                depth, metadata, parameters=self.model.parameters())

            pairs = metadata['geometry_consistency']['indices']
            pairs = pairs.cpu().numpy().tolist()

            print(f"pairs = {pairs}, loss = {loss[0]}")

            if self.params.test_2_images:
                # if abs(index1 - index2) < 10:
                #     continue
                # if (index1 == 16 and index2 == 20):
                # if (index1 == 40 and index2 == 44):
                if (index1 == 170 and index2 == 171):
                # if (index1 == 200 and index2 == 204):
                # if (index1 == 228 and index2 == 232):
                # if (index1 == 276 and index2 == 280):
                # if (index1 == 296 and index2 == 300):
                # if (index1 == 397 and index2 == 399):
                # if (index1 == 276 and index2 == 284):
                # if (index1 == 364 and index2 == 365):
                # if ~(index1 == self.params.first_image) or abs(index1 - index2) < 4:
                    break

        # pdb.set_trace()
        depth_gt_predicted_dir = pjoin(self.base_dir, 'depth_gt_predicted{}'.format(predicted_suffix))
        depth_gt_predicted_fn = pjoin(self.base_dir, 'depth_gt_predicted_list{}.json'.format(predicted_suffix))
        if True:#not os.path.isfile(depth_g1_fn):
            depth_g1_list = os.listdir(depth_gt_predicted_dir)
            depth_g1_indices = []
            for l in depth_g1_list:
                index1 = int(l.split('.')[0].split('_')[4])
                index2 = int(l.split('.')[0].split('_')[5])


                depth_g1_indices.append([index1, index2])
            import json
            with open(depth_gt_predicted_fn, 'w') as fw:
                json.dump(depth_g1_indices, fw)

        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"took {epoch_duration:.2f}s.")


    def inference(self, writer=None):
        meta_file = pjoin(self.range_dir, "metadata_scaled.npz")
        # meta_file = pjoin(self.range_dir, "metadata.npz")

        predicted_suffix = "_eps2_{}_gdt_{}_dmm_{}".format(self.params.eps2, self.params.g1_diff_threshold, self.params.depth_mean_min)

        dataset = VideoDataset(self.base_dir, meta_file, params=self.params, suffix=predicted_suffix)
        test_data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        criterion = JointLoss(self.params,
            parameters_init=[p.clone() for p in self.model.parameters()])
        criterion_l1  = torch.nn.L1Loss()

        test_dir = pjoin(self.out_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        eval_dir = pjoin(self.out_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        depth_gt_predicted_dir = pjoin(self.base_dir, 'depth_gt_predicted{}'.format(predicted_suffix))
        os.makedirs(depth_gt_predicted_dir, exist_ok=True)
        depth_gt_predicted_png_dir = pjoin(self.base_dir, 'depth_gt_predicted_png{}'.format(predicted_suffix))
        os.makedirs(depth_gt_predicted_png_dir, exist_ok=True)
        mask_rectified = pjoin(self.base_dir, 'mask_rectified{}'.format(predicted_suffix))
        os.makedirs(mask_rectified, exist_ok=True)
        points_cloud_dir = pjoin(self.base_dir, 'points_cloud')
        os.makedirs(points_cloud_dir, exist_ok=True)

        self.model.eval()
        # self.model.train()

        def suffix(epoch, niters):
            return "_e{:04d}_iter{:06d}".format(epoch, niters)

        def validate(epoch, niters):
            loss_meta = self.eval_and_save(
                criterion, test_data_loader, suffix(epoch, niters)
            )
            if writer is not None:
                log_loss_stats(
                    writer, "validation", loss_meta, epoch, log_histogram=True
                )
            print(f"Done Validation for epoch {epoch} ({niters} iterations)")

        # self.vis_depth_scale = 1.5639507 # None
        # validate(0, 0)
        # pdb.set_trace()
        # depth_g1_dir = pjoin(self.base_dir, 'depth_g1')
        # depth_g1_fn = pjoin(self.base_dir, 'depth_g1_list.json')
        # if True:#not os.path.isfile(depth_g1_fn):
        #     depth_g1_list = os.listdir(depth_g1_dir)
        #     depth_g1_indices = []
        #     for l in depth_g1_list:
        #         index1 = int(l.split('.')[0].split('_')[3])
        #         index2 = int(l.split('.')[0].split('_')[4])
        #         depth_g1_indices.append([index1, index2])
        #     import json
        #     fw = open(depth_g1_fn, 'w')
        #     json.dump(depth_g1_indices, fw)
        # pdb.set_trace()

        epoch_start_time = time.perf_counter()
        for data in test_data_loader:
            pdb.set_trace()

            index1 = data[1]['geometry_consistency']['indices'][0][0]
            index2 = data[1]['geometry_consistency']['indices'][0][1]
            if self.params.test_2_images:
                # if abs(index1 - index2) < 10:
                #     continue
                # if ~(index1 == 16 and index2 == 20):
                # if ~(index1 == 40 and index2 == 44):
                if ~(index1 == 170 and index2 == 171):
                # if ~(index1 == 200 and index2 == 204):
                # if ~(index1 == 228 and index2 == 232):
                # if ~(index1 == 276 and index2 == 280):
                # if ~(index1 == 296 and index2 == 300):
                # if ~(index1 == 397 and index2 == 399):
                # if ~(index1 == 276 and index2 == 284):
                # if ~(index1 == 364 and index2 == 365):
                # if ~(index1 == self.params.first_image) or abs(index1 - index2) < 4:
                    continue

            data = to_device(data)
            stacked_img, metadata = data
            pdb.set_trace()
            with torch.no_grad():
                depth = self.model(stacked_img, metadata)

            # pdb.set_trace()
            ## visualize depth 
            inv_depth = 1.0 / depth.cpu().detach().numpy()
            inv_depth0 = inv_depth[0, 0, ...]# * data[1]['geometry_consistency']['masks'][0].cpu().numpy()[0, 0, ...]
            inv_depth1 = inv_depth[0, 1, ...]# * data[1]['geometry_consistency']['masks'][1].cpu().numpy()[0, 0, ...]            
            if self.vis_depth_scale is None:
                self.vis_depth_scale = 1.5639507 # max(inv_depth0.max(), inv_depth1.max())
            inv_depth_vis = visualization.visualize_depth(inv_depth0, depth_min=0, depth_max=self.vis_depth_scale)
            cv2.imwrite(pjoin(test_dir, "test_depth_frame_{}.png".format(index1)), inv_depth_vis)
            inv_depth_vis = visualization.visualize_depth(inv_depth1, depth_min=0, depth_max=self.vis_depth_scale)
            cv2.imwrite(pjoin(test_dir, "test_depth_frame_{}.png".format(index2)), inv_depth_vis)

            loss, loss_meta = criterion(
                depth, metadata, parameters=self.model.parameters())

            pairs = metadata['geometry_consistency']['indices']
            pairs = pairs.cpu().numpy().tolist()

            print(f"pairs = {pairs}, loss = {loss[0]}")

            if self.params.test_2_images:
                # if abs(index1 - index2) < 10:
                #     continue
                # if (index1 == 16 and index2 == 20):
                # if (index1 == 40 and index2 == 44):
                if (index1 == 170 and index2 == 171):
                # if (index1 == 200 and index2 == 204):
                # if (index1 == 228 and index2 == 232):
                # if (index1 == 276 and index2 == 280):
                # if (index1 == 296 and index2 == 300):
                # if (index1 == 397 and index2 == 399):
                # if (index1 == 276 and index2 == 284):
                # if (index1 == 364 and index2 == 365):
                # if ~(index1 == self.params.first_image) or abs(index1 - index2) < 4:
                    break

        # pdb.set_trace()
        depth_gt_predicted_dir = pjoin(self.base_dir, 'depth_gt_predicted{}'.format(predicted_suffix))
        depth_gt_predicted_fn = pjoin(self.base_dir, 'depth_gt_predicted_list{}.json'.format(predicted_suffix))
        if True:#not os.path.isfile(depth_g1_fn):
            depth_g1_list = os.listdir(depth_gt_predicted_dir)
            depth_g1_indices = []
            for l in depth_g1_list:
                index1 = int(l.split('.')[0].split('_')[4])
                index2 = int(l.split('.')[0].split('_')[5])


                depth_g1_indices.append([index1, index2])
            import json
            with open(depth_gt_predicted_fn, 'w') as fw:
                json.dump(depth_g1_indices, fw)

        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"took {epoch_duration:.2f}s.")
        

    def ablation_median_didabled_prep(self, writer=None):
        predicted_suffix = "_eps2_{}_gdt_{}_dmm_{}".format(self.params.eps2, self.params.g1_diff_threshold, self.params.depth_mean_min)
        depth_gt_predicted_dir = pjoin(self.base_dir, 'depth_gt_predicted{}'.format(predicted_suffix))
        depth_gt_predicted_amd_fn = pjoin(self.base_dir, 'depth_gt_predicted_list{}_amd{}.json'.format(predicted_suffix, self.params.ablation_median_didabled))
        
        depth_gt_predicted_list = sorted(os.listdir(depth_gt_predicted_dir))
        depth_gt_predicted_amd_indices = []
        frame_gt_depth_dict = {}
        for l in depth_gt_predicted_list:
            frame_id = int(l.split('.')[0].split('_')[1])
            index1 = int(l.split('.')[0].split('_')[4])
            index2 = int(l.split('.')[0].split('_')[5])
            if frame_id in frame_gt_depth_dict:
                frame_gt_depth_dict[frame_id].append([index1, index2])
            else:
                frame_gt_depth_dict[frame_id] = [[index1, index2]]
        
        import json
        with open(depth_gt_predicted_amd_fn, 'w') as fw: json.dump(frame_gt_depth_dict, fw)

        ## demo for reading masks and depths for the experiment of ablation_median_didabled
        # pair = [100, 101]
        # from loaders.video_dataset import load_mask
        # from utils.image_io import load_raw_float32_image

        # self.mask_fmt = pjoin(self.base_dir, "mask_rectified{}".format(predicted_suffix), "mask_{:06d}_{:06d}.png")
        # self.depth_gt_fmt = pjoin(self.base_dir, "depth_gt_predicted{}".format(predicted_suffix), "depth_{:06d}_gt_predicted_{:06d}_{:06d}.raw") # lbz added

        # cnt_pair = 0
        # for k_ref, _ in [pair, pair[::-1]]:
        #     k_ref_candidates = frame_gt_depth_dict[k_ref]
        #     cnt = 0
        #     for k_pair in k_ref_candidates:
        #         mask = load_mask(self.mask_fmt.format(k_pair[0], k_pair[1]), channels_first=True) # (1, 288, 384)
        #         depth = torch.from_numpy(load_raw_float32_image(self.depth_gt_fmt.format(k_ref, k_pair[0], k_pair[1]))).unsqueeze(0) # (1, 1, 288, 384)
        #         if cnt == 0:
        #             mask_stack = mask
        #             depth_stack = depth
        #             cnt += 1
        #         else:
        #             mask_stack = torch.cat((mask_stack, mask), dim=0)
        #             depth_stack = torch.cat((depth_stack, depth), dim=1)
                
        #     if cnt_pair == 0:
        #         masks = [mask_stack]
        #         depths = [depth_stack]
        #         cnt_pair += 1
        #     else:
        #         masks.append(mask_stack)
        #         depths.append(depth_stack)
        # pdb.set_trace()




    def gt_prediction_grad_check(self, writer=None):
        predicted_suffix = "_eps2_{}_gdt_{}_dmm_{}".format(self.params.eps2, self.params.g1_diff_threshold, self.params.depth_mean_min)
        depth_gt_predicted_dir = pjoin(self.base_dir, 'depth_gt_predicted{}'.format(predicted_suffix))
        depth_gt_predicted_json = pjoin(self.base_dir, 'depth_gt_predicted_list{}.json'.format(predicted_suffix))
        mask_dir = pjoin(self.base_dir, 'mask')
        mask_source_edge_fmt = pjoin(self.base_dir, "mask_source_edge", "mask_{}.png") # from single-view-mpi
        color_fmt = pjoin(self.base_dir, "color_down", "frame_{}.raw")

        depth_gt_predicted_list = sorted(os.listdir(depth_gt_predicted_dir))
        frame_gt_depth_dict = {}
        for l in depth_gt_predicted_list:
            frame_id = int(l.split('.')[0].split('_')[1])
            if frame_id in frame_gt_depth_dict:
                frame_gt_depth_dict[frame_id].append(l)
            else:
                frame_gt_depth_dict[frame_id] = [l]
        
        depth_gt_predicted_grad_masked_dir = pjoin(self.base_dir, 'depth_gt_predicted{}_grad_masked'.format(predicted_suffix))
        os.makedirs(depth_gt_predicted_grad_masked_dir, exist_ok=True)

        cnt = 0
        depth_gt_final_indices = []
        for f_id in frame_gt_depth_dict.keys():
            try:
                for d_id in range(len(frame_gt_depth_dict[f_id])):
                    depth_name = frame_gt_depth_dict[f_id][d_id]
                    index1 = int(depth_name.split('.')[0].split('_')[4])
                    index2 = int(depth_name.split('.')[0].split('_')[5])
                    if f_id == index1:
                        mask_name = 'mask_{}_{}.png'.format(str(f_id).zfill(6), str(index2).zfill(6))
                    else:
                        index2 = index1
                        mask_name = 'mask_{}_{}.png'.format(str(f_id).zfill(6), str(index2).zfill(6))
                    
                    # mask_rectified = load_mask(pjoin(mask_rectified_dir, mask_name), channels_first=True).cuda()
                    
                    depth = torch.from_numpy(load_raw_float32_image(pjoin(depth_gt_predicted_dir, depth_name))).unsqueeze(3).cuda()
                    color = torch.from_numpy(load_raw_float32_image(color_fmt.format(str(f_id).zfill(6)))).unsqueeze(0).cuda()
                    mask = load_mask(pjoin(mask_dir, mask_name), channels_first=True).cuda()
                    mask_source_edge = load_mask(mask_source_edge_fmt.format(str(f_id).zfill(6)), channels_first=True).cuda()
                    depth_median_mask = depth.sum(dim=3) * mask
                    depth_median = depth_median_mask[depth_median_mask > 0].median()
                    depth_median_mask[depth_median_mask > depth_median] = 0
                    depth_median_mask[depth_median_mask > 0] = 1

                    grad_depth = gradient(depth) - 0.05
                    grad_depth[grad_depth < 0] = 0
                    depth_grad_diff = ((grad_depth / depth[:, :, :, 0]) * depth_median_mask * mask_source_edge).sum() / (depth_median_mask * mask_source_edge).sum()

                    grad_color = gradient(color) - 0.05
                    grad_color[grad_color < 0] = 0
                    color_grad_diff = ((grad_color) * depth_median_mask * mask_source_edge).sum() / (depth_median_mask * mask_source_edge).sum()
                    
                    depth_small_grad_large_depth_small = torch.ones_like(mask)
                    
                    depth_masked = depth[:, :, :, 0] * mask

                    if (depth_grad_diff > color_grad_diff) or (depth_masked[depth_masked > 0].mean() < 0.9):
                        depth_median = depth_median_mask[depth_median_mask > 0].median()
                        depth_median_mask[depth_median_mask > depth_median] = 0
                        depth_median_mask[depth_median_mask > 0] = 1

                        depth_small_grad_large_depth_small = 1 - depth_median_mask
                        
                        cnt += 1
                        print('bad_{}_{}_depth-grad-diff_{}_color-grad-diff_{}'.format(cnt, depth_name, depth_grad_diff, color_grad_diff))
                    
                        depth_grad_masked = depth[:, :, :, 0]# * mask * depth_small_grad_large_depth_small
                        inv_depth_grad_masked_vis = visualization.visualize_depth(1 / depth_grad_masked[0].cpu().numpy(), depth_min=0, depth_max=self.vis_depth_scale)
                        cv2.imwrite(depth_gt_predicted_grad_masked_dir + '/depth_{}_gt_predicted_{}_{}_grad_masked.png'.format(f_id, f_id, index2), inv_depth_grad_masked_vis * (mask * depth_small_grad_large_depth_small).permute(1, 2, 0).cpu().numpy())
                    else:
                        depth_gt_final_indices.append([f_id, index2])
            except:
                pdb.set_trace()
        with open(depth_gt_predicted_json, 'w') as fw:
            json.dump(depth_gt_final_indices, fw)

        # pdb.set_trace()
    
        # pdb.set_trace()
        # if d_id == 0:
        #     mask_sum = mask_rectified
        #     depth_masked_sum = depth * mask_rectified
        # else:
        #     mask_sum += mask_rectified
        #     depth_masked_sum += depth * mask_rectified

        # mask_clone = mask_sum.clone()
        # mask_clone[mask_clone == 0] = 1
        # depth_average = depth_masked_sum / mask_clone

        # inv_depth_vis = visualization.visualize_depth(1 / depth_average[0], depth_min=0, depth_max=self.vis_depth_scale)
        # # pdb.set_trace()
        # for d_id in range(len(frame_gt_depth_dict[f_id])):
        #     depth_name = frame_gt_depth_dict[f_id][d_id]
        #     index1 = int(depth_name.split('.')[0].split('_')[4])
        #     index2 = int(depth_name.split('.')[0].split('_')[5])
        #     if d_id != index1:
        #         mask_name = 'mask_{}_{}.png'.format(str(index1).zfill(6), str(index2).zfill(6))
        #     else:
        #         mask_name = 'mask_{}_{}.png'.format(str(index2).zfill(6), str(index1).zfill(6))
        #     mask = load_mask(pjoin(mask_rectified_dir, mask_name), channels_first=True).numpy()
        #     image_io.save_raw_float32_image(depth_gt_prediction_prepro_dir + '/depth_{}_gt_predicted_{}_{}.raw'.format(str(index1).zfill(6), str(index1).zfill(6), str(index2).zfill(6)), depth_average)
        #     cv2.imwrite(depth_gt_prediction_prepro_png_dir + '/depth_{}_gt_predicted_{}_{}_masked_average.png'.format(f_id, index1, index2), inv_depth_vis * mask.transpose((1, 2, 0)))

    def gt_prediction_prepro_average(self, writer=None):
        predicted_suffix = "_eps2_{}_gdt_{}_dmm_{}".format(self.params.eps2, self.params.g1_diff_threshold, self.params.depth_mean_min)
        depth_gt_predicted_dir = pjoin(self.base_dir, 'depth_gt_predicted{}'.format(predicted_suffix))
        depth_gt_predicted_average_json = pjoin(self.base_dir, 'depth_gt_predicted_average_list{}.json'.format(predicted_suffix))
        mask_rectified_dir = pjoin(self.base_dir, 'mask_rectified{}'.format(predicted_suffix))
        
        mask_rectified_average_dir = pjoin(self.base_dir, 'mask_rectified_average{}'.format(predicted_suffix))
        depth_gt_prediction_prepro_dir = pjoin(self.base_dir, 'depth_gt_predicted{}_prepro{}'.format(predicted_suffix, self.params.preproID))
        depth_gt_prediction_prepro_png_dir = pjoin(self.base_dir, 'depth_gt_predicted_png{}_prepro{}'.format(predicted_suffix, self.params.preproID))
        os.makedirs(mask_rectified_average_dir, exist_ok=True)
        os.makedirs(depth_gt_prediction_prepro_dir, exist_ok=True)
        os.makedirs(depth_gt_prediction_prepro_png_dir, exist_ok=True)

        depth_gt_predicted_list = sorted(os.listdir(depth_gt_predicted_dir))
        frame_gt_depth_dict = {}
        for l in depth_gt_predicted_list:
            frame_id = int(l.split('.')[0].split('_')[1])
            if frame_id in frame_gt_depth_dict:
                frame_gt_depth_dict[frame_id].append(l)
            else:
                frame_gt_depth_dict[frame_id] = [l]

        depth_gt_final_indices = []
        for f_id in frame_gt_depth_dict.keys():
            for d_id in range(len(frame_gt_depth_dict[f_id])):
                depth_name = frame_gt_depth_dict[f_id][d_id]
                index1 = int(depth_name.split('.')[0].split('_')[4])
                index2 = int(depth_name.split('.')[0].split('_')[5])

                if f_id == index1:
                    mask_name = 'mask_{}_{}.png'.format(str(f_id).zfill(6), str(index2).zfill(6))
                else:
                    index2 = index1
                    mask_name = 'mask_{}_{}.png'.format(str(f_id).zfill(6), str(index2).zfill(6))
                
                depth = torch.from_numpy(load_raw_float32_image(pjoin(depth_gt_predicted_dir, depth_name))).cuda()
                mask_rectified = load_mask(pjoin(mask_rectified_dir, mask_name), channels_first=True).cuda()
                
                if d_id == 0:
                    mask_sum = mask_rectified
                    depth_masked_sum = depth * mask_rectified
                else:
                    mask_sum += mask_rectified
                    depth_masked_sum += depth * mask_rectified
            
            mask_clone = mask_sum.clone()
            mask_clone[mask_clone == 0] = 1
            depth_average = (depth_masked_sum / mask_clone).cpu()
            mask_sum[mask_sum > 0] = 1

            cv2.imwrite(mask_rectified_average_dir + '/mask_{}.png'.format(str(f_id).zfill(6)), mask_sum.permute(1, 2, 0).cpu().numpy() * 255)
            image_io.save_raw_float32_image(depth_gt_prediction_prepro_dir + '/depth_{}_gt_predicted_average.raw'.format(str(f_id).zfill(6)), depth_average[0].cpu().numpy())
            inv_depth_vis = visualization.visualize_depth(1 / depth_average[0], depth_min=0, depth_max=self.vis_depth_scale)
            cv2.imwrite(depth_gt_prediction_prepro_png_dir + '/depth_{}_gt_predicted_average.png'.format(f_id), inv_depth_vis * mask_sum.permute(1, 2, 0).cpu().numpy())
            neighbor_found = 0
            for d_id in range(len(frame_gt_depth_dict[f_id])):
                depth_name = frame_gt_depth_dict[f_id][d_id]
                index1 = int(depth_name.split('.')[0].split('_')[4])
                index2 = int(depth_name.split('.')[0].split('_')[5])
                if f_id == index1:
                    mask_name = 'mask_{}_{}.png'.format(str(f_id).zfill(6), str(index2).zfill(6))
                else:
                    index2 = index1
                    mask_name = 'mask_{}_{}.png'.format(str(f_id).zfill(6), str(index2).zfill(6))
                mask_original = load_mask(pjoin(mask_rectified_dir, mask_name), channels_first=True).numpy()
                cv2.imwrite(depth_gt_prediction_prepro_png_dir + '/depth_{}_gt_predicted_{}_{}_masked_average.png'.format(f_id, f_id, index2), inv_depth_vis * mask_original.transpose((1, 2, 0)))

                if (neighbor_found == 0) and (f_id == index1): # save the first pair with index > index1
                    depth_gt_final_indices.append([f_id, index2])
                    neighbor_found = 1

        with open(depth_gt_predicted_average_json, 'w') as fw:
            json.dump(depth_gt_final_indices, fw)


    def gt_prediction_prepro_selection(self, writer=None):
        # selection pixels in all versions
        predicted_suffix = "_eps2_{}_gdt_{}_dmm_{}".format(self.params.eps2, self.params.g1_diff_threshold, self.params.depth_mean_min)
        depth_gt_predicted_dir = pjoin(self.base_dir, 'depth_gt_predicted{}'.format(predicted_suffix))
        depth_gt_predicted_selection_json = pjoin(self.base_dir, 'depth_gt_predicted_selection_list{}.json'.format(predicted_suffix))
        mask_rectified_dir = pjoin(self.base_dir, 'mask_rectified{}'.format(predicted_suffix))
        
        mask_rectified_selection_dir = pjoin(self.base_dir, 'mask_rectified_selection{}'.format(predicted_suffix))
        depth_gt_prediction_prepro_dir = pjoin(self.base_dir, 'depth_gt_predicted{}_prepro{}'.format(predicted_suffix, self.params.preproID))
        depth_gt_prediction_prepro_png_dir = pjoin(self.base_dir, 'depth_gt_predicted_png{}_prepro{}'.format(predicted_suffix, self.params.preproID))
        os.makedirs(mask_rectified_selection_dir, exist_ok=True)
        os.makedirs(depth_gt_prediction_prepro_dir, exist_ok=True)
        os.makedirs(depth_gt_prediction_prepro_png_dir, exist_ok=True)

        # build the dictionary, frame_gt_depth_dict, like this:
        # {frame_id: {pair_dist: {"pair_list": [pair_a, pair_b]}}}
        depth_gt_predicted_list = sorted(os.listdir(depth_gt_predicted_dir))
        frame_gt_depth_dict = {}
        for l in depth_gt_predicted_list: # l: depth_000000_gt_predicted_000000_000001.raw
            frame_id = int(l.split('.')[0].split('_')[1])
            pair_dist = abs(int(l.split('.')[0].split('_')[4]) - int(l.split('.')[0].split('_')[5]))
            if frame_id in frame_gt_depth_dict:
                if pair_dist in frame_gt_depth_dict[frame_id]:
                    frame_gt_depth_dict[frame_id][pair_dist]["pair_list"].append(l)
                else:
                    frame_gt_depth_dict[frame_id][pair_dist] = {"pair_list": [l]}
            else:
                frame_gt_depth_dict[frame_id] = {pair_dist: {"pair_list": [l]}}

        # build the dictionary, frame_gt_depth_dict, like this:
        # {frame_id: {pair_dist: {"pair_list": [pair_a, pair_b], "depth_masked_stack": depth_stack, "mask_stack": mask_stack}}}
        depth_gt_final_indices = []
        for frame_id in frame_gt_depth_dict.keys():
            dist_cnt = 0
            for pair_dist in reversed(sorted(frame_gt_depth_dict[frame_id].keys())):
                pair_list = frame_gt_depth_dict[frame_id][pair_dist]["pair_list"]
                pair_cnt = 0
                for p in pair_list:
                    index1 = int(p.split('.')[0].split('_')[4])
                    index2 = int(p.split('.')[0].split('_')[5])
                    if frame_id == index1:
                        mask_name = 'mask_{}_{}.png'.format(str(frame_id).zfill(6), str(index2).zfill(6))
                    else:
                        index2 = index1
                        mask_name = 'mask_{}_{}.png'.format(str(frame_id).zfill(6), str(index2).zfill(6))
                    depth = torch.from_numpy(load_raw_float32_image(pjoin(depth_gt_predicted_dir, p))).cuda() # 1*W*H
                    mask_rectified = load_mask(pjoin(mask_rectified_dir, mask_name), channels_first=True).cuda() # 1*W*H
                    depth_masked = depth * mask_rectified

                    if pair_cnt == 0:
                        depth_masked_stack = depth_masked
                        mask_stack = mask_rectified
                        pair_cnt += 1
                    else:
                        depth_masked_stack = torch.cat((depth_masked_stack, depth_masked), dim=0)
                        mask_stack = torch.cat((mask_stack, mask_rectified), dim=0)
                
                depth_masked_sum = torch.sum(depth_masked_stack, dim=0)
                mask_sum = torch.sum(mask_stack, dim=0)
                
                mask_clone = mask_sum.clone()
                mask_clone[mask_clone == 0] = 1
                depth_masked_average = (depth_masked_sum / mask_clone)
                mask_sum[mask_sum > 0] = 1
                frame_gt_depth_dict[frame_id][pair_dist]["depth_masked_average"] = depth_masked_average
                frame_gt_depth_dict[frame_id][pair_dist]["mask_union"] = mask_sum

                if dist_cnt == 0:
                    depth_selection = depth_masked_average
                    mask_selection = mask_sum
                    dist_cnt += 1
                else:
                    depth_selection = depth_selection + depth_masked_average * (1 - mask_selection) * mask_sum
                    mask_selection = mask_selection + (1 - mask_selection) * mask_sum

            cv2.imwrite(mask_rectified_selection_dir + '/mask_{}.png'.format(str(frame_id).zfill(6)), mask_selection.cpu().numpy() * 255)
            image_io.save_raw_float32_image(depth_gt_prediction_prepro_dir + '/depth_{}_gt_predicted_selection.raw'.format(str(frame_id).zfill(6)), depth_selection.cpu().numpy())
            inv_depth_vis = visualization.visualize_depth(1 / (depth_selection.cpu().numpy() + 1e-6), depth_min=0, depth_max=self.vis_depth_scale)
            cv2.imwrite(depth_gt_prediction_prepro_png_dir + '/depth_{}_gt_predicted_selection.png'.format(frame_id), inv_depth_vis * mask_selection.unsqueeze(2).cpu().numpy())
            
            neighbor_found = 0
            for pair_dist in sorted(frame_gt_depth_dict[frame_id].keys()):
                for depth_name in frame_gt_depth_dict[frame_id][pair_dist]["pair_list"]:
                    index1 = int(depth_name.split('.')[0].split('_')[4])
                    index2 = int(depth_name.split('.')[0].split('_')[5])
                    if frame_id == index1:
                        mask_name = 'mask_{}_{}.png'.format(str(frame_id).zfill(6), str(index2).zfill(6))
                    else:
                        index2 = index1
                        mask_name = 'mask_{}_{}.png'.format(str(frame_id).zfill(6), str(index2).zfill(6))
                    mask_original = load_mask(pjoin(mask_rectified_dir, mask_name), channels_first=True).numpy()
                    cv2.imwrite(depth_gt_prediction_prepro_png_dir + '/depth_{}_gt_predicted_{}_{}_masked_selection.png'.format(frame_id, frame_id, index2), inv_depth_vis * mask_original.transpose((1, 2, 0)))

                    if (neighbor_found == 0) and (frame_id == index1): # save the first pair with index > index1
                        depth_gt_final_indices.append([frame_id, index2])
                        neighbor_found = 1

        with open(depth_gt_predicted_selection_json, 'w') as fw:
            json.dump(depth_gt_final_indices, fw)



    def depth_projection_between_frames_selection(self, writer=None):
        import numpy as np
        import torch.nn.functional as F
        from pytorch3d.structures import Pointclouds
        from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
        from pytorch3d.renderer import (
            PerspectiveCameras, 
            PointsRasterizationSettings,
            PointsRenderer,
            PointsRasterizer,
            AlphaCompositor,
        )
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        
        predicted_suffix = "_eps2_{}_gdt_{}_dmm_{}".format(self.params.eps2, self.params.g1_diff_threshold, self.params.depth_mean_min)
        masks_dir = pjoin(self.base_dir, "mask_rectified_selection{}".format(predicted_suffix))
        depths_dir = pjoin(self.base_dir, "depth_gt_predicted{}_prepro2".format(predicted_suffix))
        depths_projection_dir = pjoin(self.base_dir, "depth_gt_predicted_png{}_prepro{}_projection_pd{}_border{}".format(predicted_suffix, self.params.preproID, self.params.projection_distance, self.params.border))
        os.makedirs(depths_projection_dir, exist_ok=True)
        depths_projection_raw_dir = pjoin(self.base_dir, "depth_gt_predicted{}_prepro{}_projection_pd{}_border{}".format(predicted_suffix, self.params.preproID, self.params.projection_distance, self.params.border))
        os.makedirs(depths_projection_raw_dir, exist_ok=True)
        mask_rectified_selection_projection_dir = pjoin(self.base_dir, 'mask_rectified_selection{}_projection_pd{}_border{}'.format(predicted_suffix, self.params.projection_distance, self.params.border))
        os.makedirs(mask_rectified_selection_projection_dir, exist_ok=True)
        depth_projection_median_confidence_map_dir = pjoin(self.base_dir, 'depth_projection_selection_confidence_map{}_ctp{}'.format(predicted_suffix, self.params.confidence_tolerance_projection))
        os.makedirs(depth_projection_median_confidence_map_dir, exist_ok=True)
        depth_fmt = pjoin(depths_dir, "depth_{:06d}_gt_predicted_selection.raw")
        mask_fmt = pjoin(masks_dir, "mask_{:06d}.png")
        # points_fmt = pjoin(points_dir, "frame_{:06d}.npz")
        meta_file = pjoin(self.range_dir, "metadata_scaled.npz")
        with open(meta_file, "rb") as f:
            meta = np.load(f)
            extrinsics = torch.tensor(meta["extrinsics"])
            intrinsics = torch.tensor(meta["intrinsics"])

        frame_info_dict = {}
        depths_list = sorted(os.listdir(depths_dir))
        for l in depths_list:
            frame_id = int(l.split('.')[0].split('_')[1])
            depth = torch.from_numpy(load_raw_float32_image(depth_fmt.format(frame_id))).to(device)
            mask = load_mask(mask_fmt.format(frame_id), channels_first=True)[0].to(device)
            pixels = pixel_grid(1, depth.shape)
            points_cam = pixels_to_points(intrinsics[frame_id].unsqueeze(0).to(device), depth.unsqueeze(0).unsqueeze(0), pixels)
            points_world = to_worldspace(points_cam, extrinsics[frame_id].unsqueeze(0).to(device))[0][0]

            # with open(points_fmt.format(frame_id), "rb") as fp:
            #     points_world = torch.tensor(np.load(fp)["points"][0]).to(device)

            frame_info_dict[frame_id] = {"depth": depth, "mask": mask, \
                "points": points_world, \
                "intrinsics": intrinsics[frame_id].unsqueeze(0), \
                "extrinsics": extrinsics[frame_id].unsqueeze(0)}
            #.permute(0, 2, 3, 1).view(-1, 3)
            # mask_cat = torch.cat((mask, mask, mask), dim=1)#.permute(0, 2, 3, 1).view(-1, 3)

            # pdb.set_trace()

        PD = self.params.projection_distance # projection distance
        for k in frame_info_dict.keys():
        # for k in range(102, 122):
            # k = 426

            depth_base = frame_info_dict[k]["depth"].unsqueeze(0).float() # 1*H*W
            # mask_base = frame_info_dict[k]["mask"].unsqueeze(0).float() # 1*H*W
            # points_base = frame_info_dict[k]["points"].float() # 3*H*W
            intrinsics_base = frame_info_dict[k]["intrinsics"].float() # 1*4
            extrinsics_base = frame_info_dict[k]["extrinsics"].float() # 1*3*4

            inv_depth_vis_base = visualization.visualize_depth(1 / (depth_base[0].cpu().numpy() + 1e-6), depth_min=0, depth_max=self.vis_depth_scale)
            cv2.imwrite(depths_projection_dir + '/depth_projection_between_frames_{:06d}.png'.format(k), inv_depth_vis_base )
            
            # depth_stack = depth_base
            cnt = 0
            for f in range(max(k-PD, 0), min(k+PD, len(frame_info_dict) + 1)):
                if f in frame_info_dict.keys():# and f != k:
                    epoch_start_time = time.perf_counter()

                    # f = 432
                    # depth = frame_info_dict[f]["depth"].unsqueeze(0).float() # 1*H*W
                    mask = frame_info_dict[f]["mask"].float().unsqueeze(0).unsqueeze(0).to(device) # 1*1*H*W
                    points = frame_info_dict[f]["points"].float().unsqueeze(0).to(device) # 1*3*H*W
                    points_to_base = to_camera(points, extrinsics_base.to(device)) # 1*3*H*W
                    points_to_base_masked = points_to_base.reshape(3, -1).permute(1, 0).contiguous()
                    verts = points_to_base_masked * torch.Tensor([-1, 1, -1]).reshape(-1, 3).contiguous().to(device) # N*3
                    depth_to_base = torch.norm(points_to_base, dim=1, keepdim=True)[0, ...] # 1*W*H
                    depth = depth_to_base.permute(1, 2, 0).reshape(-1, 1).contiguous().to(device) # N*1

                    H, W = mask.squeeze().shape
                    cameras = PerspectiveCameras(image_size=torch.tensor((W, H)).unsqueeze(0), focal_length=intrinsics_base[:, :2], principal_point=intrinsics_base[:, 2:], device=device)
                    raster_settings = PointsRasterizationSettings(image_size=mask.squeeze().shape, radius = 0.01, points_per_pixel = 10, bin_size=100)
                    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
                    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
                    point_cloud = Pointclouds(points=[verts], features=[depth])
                    try:
                        # pdb.set_trace()
                        depth_projection = renderer(point_cloud)
                    except:
                        continue
                    inv_depth_vis = visualization.visualize_depth(1 / (depth_projection.squeeze().cpu().numpy() + 1e-6), depth_min=0, depth_max=self.vis_depth_scale)
                    cv2.imwrite(depths_projection_dir + '/depth_projection_between_frames_{:06d}_{:06d}.png'.format(f, k), inv_depth_vis )
                    
                    # if self.params.upsampling_factor != 0:
                        # pdb.set_trace()

                        # scene_dir = pjoin(self.base_dir, 'color_down_png/frame_{:06d}.png'.format(k))
                        # mask_projection = torch.ones_like(depth_projection.squeeze())
                        # mask_projection[depth_projection.squeeze() < 0.1] = 0
                        # depth_projection_smoothed = fast_bilateral_solver_mask(scene_dir, depth_projection.squeeze().cpu().numpy(), mask_projection.cpu().numpy(), self.params.upsampling_factor)
                        # inv_depth_projection_smoothed_vis = visualization.visualize_depth(1 / (depth_projection_smoothed + 1e-6), depth_min=0, depth_max=self.vis_depth_scale)
                        # cv2.imwrite(depths_projection_dir + '/depth_projection_between_frames_{:06d}_{:06d}_smoothed.png'.format(f, k), inv_depth_projection_smoothed_vis * mask_projection[..., None].cpu().numpy())
                    
                    epoch_end_time = time.perf_counter()
                    epoch_duration = epoch_end_time - epoch_start_time
                    print("depth_projection_between_frames_{:06d}_{:06d} took {:.2f}s.".format(f, k, epoch_duration))
                
                    cnt+=1
                    if cnt == 1:
                        depth_stack = depth_projection
                    else:
                        depth_stack = torch.cat((depth_stack, depth_projection), 0) # 1*H*W

            depth_clone = depth_stack.clone()
            import numpy as np
            depth_clone[depth_clone == 0] = np.nan
            depth_median = torch.tensor(np.nan_to_num(np.nanmedian(depth_clone.cpu().numpy(), axis=0)))
            mask_median = torch.ones_like(depth_median)
            mask_median[depth_median == 0] = 0

            # calculate and save the confidence map of depth projection
            depth_clone = np.nan_to_num(depth_clone.cpu().numpy())
            depth_diff = np.abs(depth_clone - depth_median[None, ...].cpu().numpy())
            depth_projection_median_confidence_map = np.sum((depth_diff <= depth_median[None, ...].cpu().numpy() * self.params.confidence_tolerance_projection), axis=0)[..., 0]
            depth_projection_median_confidence_map_path = depth_projection_median_confidence_map_dir + '/frame_{}.npz'.format(str(k).zfill(6))        
            # np.savez(depth_projection_median_confidence_map_path, points=depth_projection_median_confidence_map/depth_projection_median_confidence_map.max())
            np.savez(depth_projection_median_confidence_map_path, points=depth_projection_median_confidence_map)

            inv_depth_median_vis = visualization.visualize_depth(1 / (depth_median.squeeze().cpu().numpy() + 1e-6), depth_min=0, depth_max=self.vis_depth_scale)
            cv2.imwrite(depths_projection_dir + '/depth_projection_frames_projection_median_{:06d}.png'.format(k), inv_depth_median_vis * mask_median.numpy())
            cv2.imwrite(mask_rectified_selection_projection_dir + '/mask_{}.png'.format(str(k).zfill(6)), mask_median.numpy() * 255)
            image_io.save_raw_float32_image(depths_projection_raw_dir + '/depth_projection_frames_projection_median_{:06d}.raw'.format(k), depth_median.cpu().numpy())


    def gt_prediction_prepro_median(self, writer=None):
        predicted_suffix = "_eps2_{}_gdt_{}_dmm_{}".format(self.params.eps2, self.params.g1_diff_threshold, self.params.depth_mean_min)
        depth_gt_predicted_dir = pjoin(self.base_dir, 'depth_gt_predicted{}'.format(predicted_suffix))
        depth_gt_predicted_median_json = pjoin(self.base_dir, 'depth_gt_predicted_median_list{}.json'.format(predicted_suffix))
        mask_rectified_dir = pjoin(self.base_dir, 'mask_rectified{}'.format(predicted_suffix))
        
        mask_rectified_median_dir = pjoin(self.base_dir, 'mask_rectified_median{}'.format(predicted_suffix))
        depth_gt_prediction_prepro_dir = pjoin(self.base_dir, 'depth_gt_predicted{}_prepro{}'.format(predicted_suffix, self.params.preproID))
        depth_gt_prediction_prepro_png_dir = pjoin(self.base_dir, 'depth_gt_predicted_png{}_prepro{}'.format(predicted_suffix, self.params.preproID))
        depth_median_confidence_map_dir = pjoin(self.base_dir, 'depth_median_confidence_map{}_ctm{}'.format(predicted_suffix, self.params.confidence_tolerance_median))
        depth_median_confidence_map_normalized_png_dir = pjoin(self.base_dir, 'depth_median_confidence_map{}_ctm{}_normalized_png'.format(predicted_suffix, self.params.confidence_tolerance_median))
        os.makedirs(mask_rectified_median_dir, exist_ok=True)
        os.makedirs(depth_gt_prediction_prepro_dir, exist_ok=True)
        os.makedirs(depth_gt_prediction_prepro_png_dir, exist_ok=True)
        os.makedirs(depth_median_confidence_map_dir, exist_ok=True)
        os.makedirs(depth_median_confidence_map_normalized_png_dir, exist_ok=True)

        depth_gt_predicted_list = sorted(os.listdir(depth_gt_predicted_dir))
        frame_gt_depth_dict = {}
        for l in depth_gt_predicted_list:
            frame_id = int(l.split('.')[0].split('_')[1])
            if frame_id in frame_gt_depth_dict:
                frame_gt_depth_dict[frame_id].append(l)
            else:
                frame_gt_depth_dict[frame_id] = [l]

        depth_gt_final_indices = []
        for f_id in frame_gt_depth_dict.keys():
            for d_id in range(len(frame_gt_depth_dict[f_id])):
                depth_name = frame_gt_depth_dict[f_id][d_id]
                index1 = int(depth_name.split('.')[0].split('_')[4])
                index2 = int(depth_name.split('.')[0].split('_')[5])

                if f_id == index1:
                    mask_name = 'mask_{}_{}.png'.format(str(f_id).zfill(6), str(index2).zfill(6))
                else:
                    index2 = index1
                    mask_name = 'mask_{}_{}.png'.format(str(f_id).zfill(6), str(index2).zfill(6))
                
                depth = torch.from_numpy(load_raw_float32_image(pjoin(depth_gt_predicted_dir, depth_name))).cuda()
                mask_rectified = load_mask(pjoin(mask_rectified_dir, mask_name), channels_first=True).cuda()

                # if self.params.upsampling_factor != 0:
                    # pdb.set_trace()
                    # scene_dir = pjoin(self.base_dir, 'color_down_png/frame_{:06d}.png'.format(f_id))
                    # depth_smoothed = fast_bilateral_solver_mask(scene_dir, depth.squeeze().cpu().numpy(), mask_rectified.squeeze().cpu().numpy(), self.params.upsampling_factor)
                    # inv_depth_smoothed_vis = visualization.visualize_depth(1 / (depth_smoothed + 1e-6), depth_min=0, depth_max=self.vis_depth_scale)
                    # cv2.imwrite(depth_gt_prediction_prepro_png_dir + '/depth_{}_gt_predicted_{}_{}_masked_median_smoothed.png'.format(f_id, f_id, index2), inv_depth_smoothed_vis * mask_rectified.squeeze()[..., None].cpu().numpy())

                # pdb.set_trace()
                depth_save = depth
                # depth_save = torch.from_numpy(depth_smoothed).permute(2, 0, 1).cuda()
                if d_id == 0:
                    mask_stack = mask_rectified
                    depth_stack = depth_save
                else:
                    mask_stack = torch.cat((mask_stack, mask_rectified), 0)
                    depth_stack = torch.cat((depth_stack, depth_save), 0)
            
            mask_clone = mask_stack.clone()
            depth_clone = depth_stack.clone()
            depth_clone_masked = depth_clone * mask_clone
            import numpy as np
            depth_clone_masked[depth_clone_masked == 0] = np.nan
            depth_median = np.nan_to_num(np.nanmedian(depth_clone_masked.cpu().numpy(), axis=0))
            mask = mask_stack.sum(dim=0).cpu().numpy()
            mask[mask > 0] = 1

            # pdb.set_trace()
            depth_clone_masked = np.nan_to_num(depth_clone_masked.cpu())
            depth_diff = np.abs(depth_clone_masked - depth_median[None, ...])
            depth_median_confidence_map = np.sum((depth_diff <= depth_median[None, ...] * self.params.confidence_tolerance_median), axis=0)
            depth_median_confidence_map_path = depth_median_confidence_map_dir + '/frame_{}.npz'.format(str(f_id).zfill(6))        
            # np.savez(depth_median_confidence_map_path, points=depth_median_confidence_map/depth_median_confidence_map.max())
            np.savez(depth_median_confidence_map_path, points=depth_median_confidence_map)

            cv2.imwrite(mask_rectified_median_dir + '/mask_{}.png'.format(str(f_id).zfill(6)), mask * 255)
            cv2.imwrite(depth_median_confidence_map_normalized_png_dir + '/confidence_{}.png'.format(str(f_id).zfill(6)), depth_median_confidence_map/depth_median_confidence_map.max() * mask * 255)
            image_io.save_raw_float32_image(depth_gt_prediction_prepro_dir + '/depth_{}_gt_predicted_median.raw'.format(str(f_id).zfill(6)), depth_median)
            inv_depth_vis = visualization.visualize_depth(1 / (depth_median + 1e-6), depth_min=0, depth_max=self.vis_depth_scale)
            cv2.imwrite(depth_gt_prediction_prepro_png_dir + '/depth_{}_gt_predicted_median.png'.format(f_id), inv_depth_vis * mask[:, :, np.newaxis])
            neighbor_found = 0
            for d_id in range(len(frame_gt_depth_dict[f_id])):
                depth_name = frame_gt_depth_dict[f_id][d_id]
                index1 = int(depth_name.split('.')[0].split('_')[4])
                index2 = int(depth_name.split('.')[0].split('_')[5])
                if f_id == index1:
                    mask_name = 'mask_{}_{}.png'.format(str(f_id).zfill(6), str(index2).zfill(6))
                else:
                    index2 = index1
                    mask_name = 'mask_{}_{}.png'.format(str(f_id).zfill(6), str(index2).zfill(6))
                mask_original = load_mask(pjoin(mask_rectified_dir, mask_name), channels_first=True).numpy()
                cv2.imwrite(depth_gt_prediction_prepro_png_dir + '/depth_{}_gt_predicted_{}_{}_masked_median.png'.format(f_id, f_id, index2), inv_depth_vis * mask_original.transpose((1, 2, 0)))

                if (neighbor_found == 0) and (f_id == index1): # save the first pair with index > index1
                    depth_gt_final_indices.append([f_id, index2])
                    neighbor_found = 1

        with open(depth_gt_predicted_median_json, 'w') as fw:
            json.dump(depth_gt_final_indices, fw)



    def depth_projection_between_frames(self, writer=None):
        import numpy as np
        import torch.nn.functional as F
        from pytorch3d.structures import Pointclouds
        from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
        from pytorch3d.renderer import (
            look_at_view_transform,
            PerspectiveCameras, 
            FoVOrthographicCameras,
            FoVPerspectiveCameras, 
            PointsRasterizationSettings,
            PointsRenderer,
            PulsarPointsRenderer,
            PointsRasterizer,
            AlphaCompositor,
            NormWeightedCompositor
        )
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        
        predicted_suffix = "_eps2_{}_gdt_{}_dmm_{}".format(self.params.eps2, self.params.g1_diff_threshold, self.params.depth_mean_min)
        masks_dir = pjoin(self.base_dir, "mask_rectified_median{}".format(predicted_suffix))
        points_dir = pjoin(self.base_dir, "points_cloud")
        depths_dir = pjoin(self.base_dir, "depth_gt_predicted{}_prepro10".format(predicted_suffix))
        depths_projection_dir = pjoin(self.base_dir, "depth_gt_predicted_png{}_prepro1_projection_pd{}_border{}".format(predicted_suffix, self.params.projection_distance, self.params.border))
        os.makedirs(depths_projection_dir, exist_ok=True)
        depths_projection_raw_dir = pjoin(self.base_dir, "depth_gt_predicted{}_prepro1_projection_pd{}_border{}".format(predicted_suffix, self.params.projection_distance, self.params.border))
        os.makedirs(depths_projection_raw_dir, exist_ok=True)
        depths_projection_smoothed_raw_dir = pjoin(self.base_dir, "depth_gt_predicted{}_prepro1_projection_pd{}_border{}_smoothed".format(predicted_suffix, self.params.projection_distance, self.params.border))
        os.makedirs(depths_projection_smoothed_raw_dir, exist_ok=True)
        mask_rectified_median_projection_dir = pjoin(self.base_dir, 'mask_rectified_median{}_projection_pd{}_border{}'.format(predicted_suffix, self.params.projection_distance, self.params.border))
        os.makedirs(mask_rectified_median_projection_dir, exist_ok=True)
        depth_median_confidence_map_dir = pjoin(self.base_dir, 'depth_median_confidence_map{}_ctm{}'.format(predicted_suffix, self.params.confidence_tolerance_median))
        depth_projection_median_confidence_map_dir = pjoin(self.base_dir, 'depth_projection_median_confidence_map{}_ctm{}_ctp{}'.format(predicted_suffix, self.params.confidence_tolerance_median, self.params.confidence_tolerance_projection))
        os.makedirs(depth_projection_median_confidence_map_dir, exist_ok=True)
        depth_fmt = pjoin(depths_dir, "depth_{:06d}_gt_predicted_median.raw")
        mask_fmt = pjoin(masks_dir, "mask_{:06d}.png")
        # points_fmt = pjoin(points_dir, "frame_{:06d}.npz")
        depth_median_confidence_map_fmt = pjoin(depth_median_confidence_map_dir, "frame_{:06d}.npz")
        meta_file = pjoin(self.range_dir, "metadata_scaled.npz")
        with open(meta_file, "rb") as f:
            meta = np.load(f)
            extrinsics = torch.tensor(meta["extrinsics"])
            intrinsics = torch.tensor(meta["intrinsics"])

        frame_info_dict = {}
        depths_list = sorted(os.listdir(depths_dir))
        for l in depths_list:
            frame_id = int(l.split('.')[0].split('_')[1])
            depth = torch.from_numpy(load_raw_float32_image(depth_fmt.format(frame_id))).to(device)
            mask = load_mask(mask_fmt.format(frame_id), channels_first=True)[0].to(device)
            pixels = pixel_grid(1, depth.shape)
            points_cam = pixels_to_points(intrinsics[frame_id].unsqueeze(0).to(device), depth.unsqueeze(0).unsqueeze(0), pixels)
            points_world = to_worldspace(points_cam, extrinsics[frame_id].unsqueeze(0).to(device))[0][0]

            # with open(points_fmt.format(frame_id), "rb") as fp:
            #     points_world = torch.tensor(np.load(fp)["points"][0]).to(device)
            with open(depth_median_confidence_map_fmt.format(frame_id), "rb") as fp:
                depth_median_confidence_map = torch.tensor(np.load(fp)["points"]).to(device)

            frame_info_dict[frame_id] = {"depth": depth, "mask": mask, \
                "points": points_world, \
                "depth_median_confidence_map": depth_median_confidence_map, \
                "intrinsics": intrinsics[frame_id].unsqueeze(0), \
                "extrinsics": extrinsics[frame_id].unsqueeze(0)}
            #.permute(0, 2, 3, 1).view(-1, 3)
            # mask_cat = torch.cat((mask, mask, mask), dim=1)#.permute(0, 2, 3, 1).view(-1, 3)

            # pdb.set_trace()

        PD = self.params.projection_distance # projection distance
        for k in frame_info_dict.keys():
        # for k in range(102, 122):
            # k = 426

            depth_base = frame_info_dict[k]["depth"].unsqueeze(0).float() # 1*H*W
            # mask_base = frame_info_dict[k]["mask"].unsqueeze(0).float() # 1*H*W
            # points_base = frame_info_dict[k]["points"].float() # 3*H*W
            intrinsics_base = frame_info_dict[k]["intrinsics"].float() # 1*4
            extrinsics_base = frame_info_dict[k]["extrinsics"].float() # 1*3*4

            inv_depth_vis_base = visualization.visualize_depth(1 / (depth_base[0].cpu().numpy() + 1e-6), depth_min=0, depth_max=self.vis_depth_scale)
            cv2.imwrite(depths_projection_dir + '/depth_projection_between_frames_{:06d}.png'.format(k), inv_depth_vis_base )
            
            # depth_stack = depth_base
            cnt = 0
            for f in range(max(k-PD, 0), min(k+PD, len(frame_info_dict) + 1)):
                if f in frame_info_dict.keys():# and f != k:
                    epoch_start_time = time.perf_counter()

                    # f = 432
                    # depth = frame_info_dict[f]["depth"].unsqueeze(0).float() # 1*H*W
                    mask = frame_info_dict[f]["mask"].float().unsqueeze(0).unsqueeze(0).to(device) # 1*1*H*W
                    points = frame_info_dict[f]["points"].float().unsqueeze(0).to(device) # 1*3*H*W
                    depth_median_confidence_map = frame_info_dict[f]["depth_median_confidence_map"].float().unsqueeze(0).to(device) # 1*H*W
                    points_to_base = to_camera(points, extrinsics_base.to(device)) # 1*3*H*W
                    mask_points = torch.cat((mask, mask, mask), dim=1)
                    points_to_base_masked = points_to_base[mask_points > 0].reshape(3, -1).permute(1, 0).contiguous()
                    points_to_base_masked = points_to_base.reshape(3, -1).permute(1, 0).contiguous()
                    verts = points_to_base_masked * torch.Tensor([-1, 1, -1]).reshape(-1, 3).contiguous().to(device) # N*3
                    depth_to_base = torch.norm(points_to_base, dim=1, keepdim=True)[0, ...] # 1*H*W
                    depth_conf = torch.cat((depth_to_base, depth_median_confidence_map), dim=0).permute(1, 2, 0) # H*W*2
                    # depth_to_base_masked = depth_to_base[mask[0, ...] > 0].reshape(-1, 1).contiguous().to(device) # N*1
                    depth_to_base_masked = depth_conf.reshape(-1, 2).contiguous().to(device) # N*1
                    depth_conf = depth_to_base_masked

                    H, W = mask.squeeze().shape
                    cameras = PerspectiveCameras(image_size=torch.tensor((W, H)).unsqueeze(0), focal_length=intrinsics_base[:, :2], principal_point=intrinsics_base[:, 2:], device=device)
                    raster_settings = PointsRasterizationSettings(image_size=mask.squeeze().shape, radius = 0.01, points_per_pixel = 10, bin_size=100)
                    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
                    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
                    point_cloud = Pointclouds(points=[verts], features=[depth_conf])
                    try:
                        # pdb.set_trace()
                        depth_conf_projection = renderer(point_cloud)
                        depth_projection, conf_projection = depth_conf_projection[..., 0], depth_conf_projection[..., 1]
                    except:
                        continue
                    inv_depth_vis = visualization.visualize_depth(1 / (depth_projection.squeeze().cpu().numpy() + 1e-6), depth_min=0, depth_max=self.vis_depth_scale)
                    cv2.imwrite(depths_projection_dir + '/depth_projection_between_frames_{:06d}_{:06d}.png'.format(f, k), inv_depth_vis )
                    
                    # if self.params.upsampling_factor != 0:
                        # pdb.set_trace()

                        # scene_dir = pjoin(self.base_dir, 'color_down_png/frame_{:06d}.png'.format(k))
                        # mask_projection = torch.ones_like(depth_projection.squeeze())
                        # mask_projection[depth_projection.squeeze() < 0.1] = 0
                        # depth_projection_smoothed = fast_bilateral_solver_mask(scene_dir, depth_projection.squeeze().cpu().numpy(), mask_projection.cpu().numpy(), self.params.upsampling_factor)
                        # inv_depth_projection_smoothed_vis = visualization.visualize_depth(1 / (depth_projection_smoothed + 1e-6), depth_min=0, depth_max=self.vis_depth_scale)
                        # cv2.imwrite(depths_projection_dir + '/depth_projection_between_frames_{:06d}_{:06d}_smoothed.png'.format(f, k), inv_depth_projection_smoothed_vis * mask_projection[..., None].cpu().numpy())
                    
                    epoch_end_time = time.perf_counter()
                    epoch_duration = epoch_end_time - epoch_start_time
                    print("depth_projection_between_frames_{:06d}_{:06d} took {:.2f}s.".format(f, k, epoch_duration))
                
                    cnt+=1
                    if cnt == 1:
                        depth_stack = depth_projection
                        depth_median_confidence_map_stack = conf_projection
                    else:
                        depth_stack = torch.cat((depth_stack, depth_projection), 0) # 1*H*W
                        depth_median_confidence_map_stack = torch.cat((depth_median_confidence_map_stack, conf_projection), 0)

            depth_clone = depth_stack.clone()
            import numpy as np
            depth_clone[depth_clone == 0] = np.nan
            depth_median = torch.tensor(np.nan_to_num(np.nanmedian(depth_clone.cpu().numpy(), axis=0)))
            mask_median = torch.ones_like(depth_median)
            mask_median[depth_median == 0] = 0

            # pdb.set_trace()

            # calculate and save the confidence map of depth projection
            depth_clone = np.nan_to_num(depth_clone.cpu().numpy())
            depth_diff = np.abs(depth_clone - depth_median[None, ...].cpu().numpy())
            depth_projection_median_confidence_map = np.sum((depth_diff <= depth_median[None, ...].cpu().numpy() * self.params.confidence_tolerance_projection) * depth_median_confidence_map_stack.cpu().numpy(), axis=0)
            depth_projection_median_confidence_map_path = depth_projection_median_confidence_map_dir + '/frame_{}.npz'.format(str(k).zfill(6))        
            # np.savez(depth_projection_median_confidence_map_path, points=depth_projection_median_confidence_map/depth_projection_median_confidence_map.max())
            np.savez(depth_projection_median_confidence_map_path, points=depth_projection_median_confidence_map)


            inv_depth_median_vis = visualization.visualize_depth(1 / (depth_median.squeeze().cpu().numpy() + 1e-6), depth_min=0, depth_max=self.vis_depth_scale)
            cv2.imwrite(depths_projection_dir + '/depth_projection_frames_projection_median_{:06d}.png'.format(k), inv_depth_median_vis * mask_median.unsqueeze(2).numpy())
            # pdb.set_trace()

            cv2.imwrite(mask_rectified_median_projection_dir + '/mask_{}.png'.format(str(k).zfill(6)), mask_median.unsqueeze(2).numpy() * 255)
            image_io.save_raw_float32_image(depths_projection_raw_dir + '/depth_projection_frames_projection_median_{:06d}.raw'.format(k), depth_median.cpu().numpy())

            if self.params.upsampling_factor != 0:
                # pdb.set_trace()
                scene_dir = pjoin(self.base_dir, 'color_down_png/frame_{:06d}.png'.format(k))
                target_dir = pjoin(depths_projection_raw_dir, 'depth_projection_frames_projection_median_{:06d}.raw'.format(k))
                depth_median_smoothed = fast_bilateral_solver(scene_dir, target_dir, self.params.upsampling_factor)
                # pdb.set_trace()
                inv_depth_median_smoothed_vis = visualization.visualize_depth(1 / (depth_median_smoothed + 1e-6), depth_min=0, depth_max=self.vis_depth_scale)
                cv2.imwrite(depths_projection_dir + '/depth_projection_frames_projection_median_smoothed_{:06d}.png'.format(k), inv_depth_median_smoothed_vis * mask_median.unsqueeze(2).numpy())
                image_io.save_raw_float32_image(depths_projection_smoothed_raw_dir + '/depth_projection_frames_projection_median_{:06d}_smoothed.raw'.format(k), depth_median_smoothed)
