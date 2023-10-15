#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import os
from os.path import join as pjoin
import shutil

from depth_fine_tuning import DepthFineTuner
from flow import Flow
from scale_calibration import calibrate_scale, calibrate_scale_scannet, calibrate_scale_tum, depth_gt_npz2raw_kitti
from tools import make_video as mkvid
from utils.frame_range import FrameRange, OptionalSet
from utils.helpers import print_banner, print_title
from video import (Video, sample_pairs, sample_pairs_2)

from tum_associate import *
from utils.helpers import mkdir_ifnotexists
from tum.evaluate_rpe import transform44

import numpy as np
import random
import torch
import cv2
import pdb
import time


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DatasetProcessor:
    def __init__(self, writer=None):
        self.writer = writer

    def create_output_path(self, params):
        range_tag = f"R{params.frame_range.name}"
        flow_ops_tag = "-".join(params.flow_ops)
        name = f"{range_tag}_{flow_ops_tag}_{params.model_type}"

        out_dir = pjoin(self.path, name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def extract_frames(self, params):
        print_banner("Extracting PTS")
        self.video.extract_pts()

        print_banner("Extracting frames")
        self.video.extract_frames()

    def tum_scene_process(self, params):
        rgb_list = read_file_list(pjoin(self.path, 'tum_scene/rgb.txt'))
        depth_list = read_file_list(pjoin(self.path, 'tum_scene/depth.txt'))
        camera_list = read_file_list(pjoin(self.path, 'tum_scene/groundtruth.txt'))

        rgb_keys = list(rgb_list)
        subsample_rate = 5
        for i in range(len(rgb_keys)):
            if i % subsample_rate != 0:
                rgb_list.pop(rgb_keys[i], None)
        rgb_depth_camera_matches = associate(rgb_list, depth_list, camera_list, float(0), float(0.02))
        
        ## qvec2rotmat() is borrowed from colmap.
        def qvec2rotmat(qvec):
            return np.array([
                [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
                [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
                [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
        ROT_COLMAP_TO_NORMAL = np.diag([1, -1, -1])
        # ROT_COLMAP_TO_NORMAL = np.diag([1, 1, -1])
        def tum_camera_2_extrinsics(camera):
            tc = np.array(camera[:3], dtype=np.float)
            # Rc = qvec2rotmat(np.array(camera[3:], dtype=np.float))
            # TUM:      tx, ty, tz, qx, qy, qz, qw
            # COLMAP:   qw, qx, qy, qz, tx, ty, tz
            Rc = qvec2rotmat(np.array([camera[-1]] + camera[3:-1], dtype=np.float))
            # R, t = Rc.T, -Rc.T.dot(tc.reshape(-1, 1))
            R, t = Rc, tc.reshape(-1, 1)
            R = ROT_COLMAP_TO_NORMAL.dot(R).dot(ROT_COLMAP_TO_NORMAL.T)
            # R = ROT_COLMAP_TO_NORMAL.dot(R)
            t = ROT_COLMAP_TO_NORMAL.dot(t)
            extrinsics = np.concatenate([R, t], axis=1)
            # pdb.set_trace()
            return extrinsics

        frame_dir = "%s/tum_color_full" % self.path
        depth_dir = "%s/tum_depth_full" % self.path
        mkdir_ifnotexists(frame_dir)
        mkdir_ifnotexists(depth_dir)

        tum_dir = pjoin(self.path, 'tum_scene')
        intrinsics, extrinsics = [], []
        for i in range(len(rgb_depth_camera_matches)):
            rgb, depth, camera = rgb_depth_camera_matches[i]
            os.system('cp {} {}'.format(pjoin(tum_dir, rgb_list[rgb][0]), pjoin(frame_dir, 'frame_{}.png'.format(str(i).zfill(6)))))
            os.system('cp {} {}'.format(pjoin(tum_dir, depth_list[depth][0]), pjoin(depth_dir, 'depth_{}.png'.format(str(i).zfill(6)))))
            extrinsics.append(tum_camera_2_extrinsics(camera_list[camera]))

            # extrinsics.append(transform44(np.array([0] + camera_list[camera]).astype(np.float))[:-1, :])
            # pdb.set_trace()
            if params.freiburg_group == 1:
                # intrinsics.append(np.array([517.3, 516.5, 318.6, 255.3], dtype=np.float))
                intrinsics.append(np.array([525.0, 525.0, 319.5, 239.5], dtype=np.float))
            if params.freiburg_group == 2:
                # intrinsics.append(np.array([520.9, 521.0, 325.1, 249.7], dtype=np.float))
                intrinsics.append(np.array([525.0, 525.0, 319.5, 239.5], dtype=np.float))
            if params.freiburg_group == 3:
                # intrinsics.append(np.array([535.4, 539.2, 320.1, 247.6], dtype=np.float))
                intrinsics.append(np.array([525.0, 525.0, 319.5, 239.5], dtype=np.float))
            if params.freiburg_group == 4:
                # intrinsics.append(np.array([480, 640, 247.6, 320.1], dtype=np.float))
                intrinsics.append(np.array([525.0, 525.0, 319.5, 239.5], dtype=np.float))
        
        # pdb.set_trace()
        # if not os.path.isfile(params.video_file):
        #     os.system('ffmpeg -f image2 -r 30 -i {}/%*.png -vcodec libx264 -profile:v high444 -refs 16 -crf 10 -preset ultrafast {}'.format(frame_dir, params.video_file))
        src_meta_file = pjoin(self.out_dir, 'metadata_intrinsics_raw.npz')
        np.savez(src_meta_file, intrinsics=np.stack(intrinsics, axis=0), extrinsics=np.stack(extrinsics, axis=0))

        valid_frames = {int(s) for s in range(len(rgb_depth_camera_matches))}
        return valid_frames

    def tum_scene_intrinsics_proc(self, params):
        src_meta_file_raw = pjoin(self.out_dir, 'metadata_intrinsics_raw.npz')
        src_meta_file = pjoin(self.out_dir, 'metadata.npz')
        with open(src_meta_file_raw, "rb") as f:
            meta = np.load(f)
            extrinsics = meta["extrinsics"]
            intrinsics_raw = meta["intrinsics"]
    
        size_new = cv2.imread(pjoin(params.path, "color_down_png", "frame_{:06d}.png".format(0)), cv2.IMREAD_UNCHANGED).shape[:2][::-1]
        size_old = cv2.imread(pjoin(params.path, "color_full", "frame_{:06d}.png".format(0)), cv2.IMREAD_UNCHANGED).shape[:2][::-1]
        intrinsics = []
        for i in range(intrinsics_raw.shape[0]):
            ratio = np.array(size_new) / np.array((size_old[0], size_old[1]))
            # pdb.set_trace()
            fxy = intrinsics_raw[i][:2] * ratio
            cxy = intrinsics_raw[i][2:] * ratio
            intrinsics.append(np.concatenate((fxy, cxy)))

        np.savez(src_meta_file, intrinsics=np.stack(intrinsics, axis=0), extrinsics=extrinsics)


    def scannet_scene_process(self, params):
        """prepare depth maps, and extrinsics/intrinsics of ScanNet dataset in CVD-style.
        """

        # prepre depth maps

        depth_dir = "%s/scannet_depth_full" % self.path
        mkdir_ifnotexists(depth_dir)
        depth_list = os.listdir(depth_dir)

        scannet_depth_dir = pjoin(self.path, "scannet/depth")
        scannet_depth_list = sorted(os.listdir(scannet_depth_dir))

        if len(depth_list) == len(scannet_depth_list):
            pass
        else:
            for scannet_depth in scannet_depth_list:
                scannet_depth_path = pjoin(scannet_depth_dir, scannet_depth)
                os.system('cp {} {}'.format(scannet_depth_path, pjoin(depth_dir, 'frame_{}.png'.format(str(scannet_depth.split(".")[0]).zfill(6)))))

        # prepare extrinsics, and intrinsics

        scannet_pose_dir = pjoin(self.path, "scannet/pose")
        scannet_pose_list = sorted(os.listdir(scannet_pose_dir))
        scannet_intrinsics_path = pjoin(self.path, "scannet/intrinsic/intrinsic_depth.txt")

        def scannet_pose_2_extrinsics(pose):
            R, t = pose[:3, :3], pose[:3, 3].reshape(-1, 1)
            ROT_COLMAP_TO_NORMAL = np.array([[1, 0, 0], [0, 0, -1],[0, 1, 0]])

            R = ROT_COLMAP_TO_NORMAL.dot(R).dot(ROT_COLMAP_TO_NORMAL.T)
            t = ROT_COLMAP_TO_NORMAL.dot(t)

            extrinsics = np.concatenate([R, t], axis=1)
            return extrinsics

        extrinsics, intrinsics = [], []
        intrinsics_mat = np.loadtxt(scannet_intrinsics_path)
        for scannet_pose in scannet_pose_list:
            scannet_pose_path = pjoin(scannet_pose_dir, scannet_pose)
            pose_mat = np.loadtxt(scannet_pose_path)
            extrinsics.append(scannet_pose_2_extrinsics(pose_mat))
            intrinsics.append(np.array([intrinsics_mat[0, 0], intrinsics_mat[1, 1], intrinsics_mat[0, 2], intrinsics_mat[1, 2]], dtype=np.float))
            # intrinsics.append(np.array([577.870605, 577.870605, 319.5, 239.5], dtype=np.float))
        
        src_meta_file = pjoin(self.out_dir, 'metadata_intrinsics_raw.npz')
        np.savez(src_meta_file, intrinsics=np.stack(intrinsics, axis=0), extrinsics=np.stack(extrinsics, axis=0))

        # pdb.set_trace()

        # valid_frames = {int(s) for s in range(len(scannet_pose_list))}
        # return valid_frames


    def pipeline(self, params):
        if params.is_tum_scene:
            valid_frames = self.tum_scene_process(params)
        self.extract_frames(params)

        print_banner("Downscaling frames (raw)")
        self.video.downscale_frames("color_down", params.size, "raw")

        print_banner("Downscaling frames (png)")
        self.video.downscale_frames("color_down_png", params.size, "png")

        print_banner("Downscaling frames (for flow)")
        self.video.downscale_frames("color_flow", Flow.max_size(), "png", align=64)

        frame_range = FrameRange(
            frame_range=params.frame_range.set, num_frames=self.video.frame_count,
        )
        frames = frame_range.frames()

        print_banner("Compute initial depth")

        ft = DepthFineTuner(self.out_dir, frames, params)
        initial_depth_dir = pjoin(self.path, f"depth_{params.model_type}")
        if not self.video.check_frames(pjoin(initial_depth_dir, "depth"), "raw"):
            ft.save_depth(initial_depth_dir)

        if not params.is_tum_scene and not params.is_scannet_scene:
            valid_frames = calibrate_scale(self.video, self.out_dir, frame_range, params)
        else:
            if params.is_tum_scene:
                self.tum_scene_intrinsics_proc(params)
                valid_frames = calibrate_scale_tum(self.video, self.out_dir, frame_range, params)
            
            if params.is_scannet_scene:
                # pdb.set_trace()
                self.scannet_scene_process(params)
                self.tum_scene_intrinsics_proc(params) # sharing the same function with tum rgbd dataset
                valid_frames = calibrate_scale_scannet(self.video, self.out_dir, frame_range, params)

        # if params.is_kitti_scene:
        #     depth_gt_npz2raw_kitti(self.video, self.out_dir, frame_range, params)

        # frame range for finetuning:
        ft_frame_range = frame_range.intersection(OptionalSet(set(valid_frames)))
        print("Filtered out frames",
            sorted(set(frame_range.frames()) - set(ft_frame_range.frames())))

        print_banner("Compute flow")

        frame_pairs = sample_pairs(ft_frame_range, params.flow_ops)
        # frame_pairs = sorted(sample_pairs(ft_frame_range, params.flow_ops))
        # frame_pairs = sample_pairs_2(self.out_dir, params.pos_dist_min, ft_frame_range, params.flow_ops)
        # pdb.set_trace()

        self.flow.compute_flow(frame_pairs, params.flow_checkpoint)

        print_banner("Compute flow masks")

        self.flow.mask_valid_correspondences()

        self.flow.mask_source_edge()

        flow_list_path = self.flow.check_good_flow_pairs(
            frame_pairs, params.overlap_ratio, frame_min=params.frame_min, frame_max=params.frame_max
        )
        shutil.copyfile(flow_list_path, pjoin(self.path, "flow_list.json"))

        print_banner("Visualize flow")

        self.flow.visualize_flow(warp=True)

        print_banner("Fine-tuning")
        
        if params.test_mode:
            ft.test(writer=self.writer)
        else:

            prep_start_time = time.perf_counter()
            
            if params.straight_line_method:
                if not params.gt_predicted_is_ready:
                    ft.test(writer=self.writer)
                    params.gt_predicted_is_ready = True
            if params.gt_prediction_prepro and params.preproID == 0 and not params.prepro_is_ready: # average
                ft.gt_prediction_prepro_average(writer=self.writer)
                params.prepro_is_ready = True
            if params.gt_prediction_prepro and params.preproID == 1 and not params.prepro_is_ready: # median
                ft.gt_prediction_prepro_median(writer=self.writer)
                params.prepro_is_ready = True
            if params.gt_prediction_prepro and params.preproID == 2 and not params.prepro_is_ready: # selection
                ft.gt_prediction_prepro_selection(writer=self.writer)
                ft.depth_projection_between_frames_selection(writer=self.writer)
                params.prepro_is_ready = True
            if params.gt_prediction_prepro and params.preproID == 10 and not params.prepro_is_ready: # depth projection
                ft.gt_prediction_prepro_median(writer=self.writer)
                ft.depth_projection_between_frames(writer=self.writer)
                params.prepro_is_ready = True

            if params.gt_prediction_grad_check:
                ft.gt_prediction_grad_check(writer=self.writer)

            if params.ablation_median_didabled:
                ft.ablation_median_didabled_prep(writer=self.writer)


            prep_end_time = time.perf_counter()
            prep_duration = prep_end_time - prep_start_time
            print(f"timing--prep took {prep_duration:.2f}s.")


            ft.fine_tune(writer=self.writer)

        print_banner("Compute final depth")

        if not self.video.check_frames(pjoin(ft.out_dir, "depth"), "raw", frames):
            ft.save_depth(ft.out_dir, frames)

        if params.make_video:
            print_banner("Export visualization videos")
            self.make_videos(params, ft.out_dir)

        return initial_depth_dir, ft.out_dir, frame_range.frames()

    def process(self, params):
        set_random_seed(0)
        self.path = params.path
        os.makedirs(self.path, exist_ok=True)

        self.video_file = params.video_file

        self.out_dir = self.create_output_path(params)

        self.video = Video(params.path, params.video_file)
        self.flow = Flow(params.path, self.out_dir)

        print_title(f"Processing dataset '{self.path}'")

        print(f"Output directory: {self.out_dir}")

        if params.op == "all":
            return self.pipeline(params)
        elif params.op == "extract_frames":
            return self.extract_frames(params)
        else:
            raise RuntimeError("Invalid operation specified.")

    def make_videos(self, params, ft_depth_dir):
        args = [
            "--color_dir", pjoin(self.path, "color_down_png"),
            "--out_dir", pjoin(self.out_dir, "videos"),
            "--depth_dirs",
            pjoin(self.path, f"depth_{params.model_type}"),
            pjoin(self.path, "depth_colmap_dense"),
            pjoin(ft_depth_dir, "depth"),
        ]
        gt_dir = pjoin(self.path, "depth_gt")
        if os.path.isdir(gt_dir):
            args.append(gt_dir)

        vid_params = mkvid.MakeVideoParams().parser.parse_args(
            args,
            namespace=params
        )
        logging.info("Make videos {}".format(vid_params))
        mkvid.main(vid_params)


    def inference(self, params):
        self.path = params.path
        self.out_dir = self.create_output_path(params)

        color_full_dir = os.path.join(self.path, 'color_full')
        color_full_list = sorted(os.listdir(color_full_dir))
        frames = [i for i in range(len(color_full_list))]
        
        inference_base_dir = os.path.join(self.path, 'inference')
        os.makedirs(inference_base_dir, exist_ok=True)

        ft = DepthFineTuner(self.out_dir, frames, params)
        ft.inference_depth(inference_base_dir)




    def pipeline_nerf(self, params):        
        self.extract_frames(params)

        frame_range = FrameRange(
            frame_range=params.frame_range.set, num_frames=self.video.frame_count,
        )
        frames = frame_range.frames()

        print_banner("Compute initial depth")

        ft = DepthFineTuner(self.out_dir, frames, params)
        initial_depth_dir = pjoin(self.path, f"depth_{params.model_type}")
        if not self.video.check_frames(pjoin(initial_depth_dir, "depth"), "raw"):
            ft.save_depth(initial_depth_dir)

        if not params.is_tum_scene and not params.is_scannet_scene:
            valid_frames = calibrate_scale(self.video, self.out_dir, frame_range, params)
        else:
            if params.is_tum_scene:
                self.tum_scene_intrinsics_proc(params)
                valid_frames = calibrate_scale_tum(self.video, self.out_dir, frame_range, params)
            
            if params.is_scannet_scene:
                # pdb.set_trace()
                self.scannet_scene_process(params)
                self.tum_scene_intrinsics_proc(params) # sharing the same function with tum rgbd dataset
                valid_frames = calibrate_scale_scannet(self.video, self.out_dir, frame_range, params)

        # if params.is_kitti_scene:
        #     depth_gt_npz2raw_kitti(self.video, self.out_dir, frame_range, params)

        # frame range for finetuning:
        ft_frame_range = frame_range.intersection(OptionalSet(set(valid_frames)))
        print("Filtered out frames",
            sorted(set(frame_range.frames()) - set(ft_frame_range.frames())))

        print_banner("Compute flow")

        frame_pairs = sample_pairs(ft_frame_range, params.flow_ops)
        # frame_pairs = sorted(sample_pairs(ft_frame_range, params.flow_ops))
        # frame_pairs = sample_pairs_2(self.out_dir, params.pos_dist_min, ft_frame_range, params.flow_ops)
        # pdb.set_trace()

        self.flow.compute_flow(frame_pairs, params.flow_checkpoint)

        print_banner("Compute flow masks")

        self.flow.mask_valid_correspondences()

        self.flow.mask_source_edge()

        flow_list_path = self.flow.check_good_flow_pairs(
            frame_pairs, params.overlap_ratio, frame_min=params.frame_min, frame_max=params.frame_max
        )
        shutil.copyfile(flow_list_path, pjoin(self.path, "flow_list.json"))

        print_banner("Visualize flow")

        self.flow.visualize_flow(warp=True)

        print_banner("Fine-tuning")
        
        if params.test_mode:
            ft.test(writer=self.writer)
        else:

            prep_start_time = time.perf_counter()

            if params.straight_line_method:
                if not params.gt_predicted_is_ready:
                    ft.test(writer=self.writer)
                    # pdb.set_trace()
                    params.gt_predicted_is_ready = True
            if params.gt_prediction_prepro and params.preproID == 0 and not params.prepro_is_ready: # average
                ft.gt_prediction_prepro_average(writer=self.writer)
                params.prepro_is_ready = True
            if params.gt_prediction_prepro and params.preproID == 1 and not params.prepro_is_ready: # median
                ft.gt_prediction_prepro_median(writer=self.writer)
                params.prepro_is_ready = True
            if params.gt_prediction_prepro and params.preproID == 2 and not params.prepro_is_ready: # selection
                ft.gt_prediction_prepro_selection(writer=self.writer)
                ft.depth_projection_between_frames_selection(writer=self.writer)
                params.prepro_is_ready = True
            if params.gt_prediction_prepro and params.preproID == 10 and not params.prepro_is_ready: # depth projection
                ft.gt_prediction_prepro_median(writer=self.writer)
                ft.depth_projection_between_frames(writer=self.writer)
                params.prepro_is_ready = True

            if params.gt_prediction_grad_check:
                ft.gt_prediction_grad_check(writer=self.writer)

            if params.ablation_median_didabled:
                ft.ablation_median_didabled_prep(writer=self.writer)


            prep_end_time = time.perf_counter()
            prep_duration = prep_end_time - prep_start_time
            print(f"timing--prep took {prep_duration:.2f}s.")


            ft.fine_tune(writer=self.writer)

        print_banner("Compute final depth")

        if not self.video.check_frames(pjoin(ft.out_dir, "depth"), "raw", frames):
            ft.save_depth(ft.out_dir, frames)

        if params.make_video:
            print_banner("Export visualization videos")
            self.make_videos(params, ft.out_dir)

        return initial_depth_dir, ft.out_dir, frame_range.frames()


    def process_nerf(self, params):
        set_random_seed(0)
        self.path = params.path
        os.makedirs(self.path, exist_ok=True)

        self.out_dir = self.create_output_path(params)

        self.flow = Flow(params.path, self.out_dir)

        print_title(f"Processing dataset '{self.path}'")

        print(f"Output directory: {self.out_dir}")

        return self.pipeline_nerf(params)