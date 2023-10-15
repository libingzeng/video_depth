#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import cv2
from os.path import join as pjoin
import json
import math
import numpy as np
import torch.utils.data as data
import torch
from typing import Optional

from utils import image_io, frame_sampling as sampling

from utils.image_io import load_raw_float32_image
import pdb

_dtype = torch.float32


def load_image(
    path: str,
    channels_first: bool,
    check_channels: Optional[int] = None,
    post_proc_raw=lambda x: x,
    post_proc_other=lambda x: x,
) -> torch.FloatTensor:
    if os.path.splitext(path)[-1] == ".raw":
        im = image_io.load_raw_float32_image(path)
        im = post_proc_raw(im)
    else:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        im = post_proc_other(im)
    im = im.reshape(im.shape[:2] + (-1,))

    if check_channels is not None:
        assert (
            im.shape[-1] == check_channels
        ), "receive image of shape {} whose #channels != {}".format(
            im.shape, check_channels
        )

    if channels_first:
        im = im.transpose((2, 0, 1))
    # to torch
    return torch.tensor(im, dtype=_dtype)


def load_color(path: str, channels_first: bool) -> torch.FloatTensor:
    """
    Returns:
        torch.tensor. color in range [0, 1]
    """
    im = load_image(
        path,
        channels_first,
        post_proc_raw=lambda im: im[..., [2, 1, 0]] if im.ndim == 3 else im,
        post_proc_other=lambda im: im / 255,
    )
    return im


def load_flow(path: str, channels_first: bool) -> torch.FloatTensor:
    """
    Returns:
        flow tensor in pixels.
    """
    flow = load_image(path, channels_first, check_channels=2)
    return flow


def load_mask(path: str, channels_first: bool) -> torch.ByteTensor:
    """
    Returns:
        mask takes value 0 or 1
    """
    mask = load_image(path, channels_first, check_channels=1) > 0
    return mask.to(_dtype)


class VideoDataset(data.Dataset):
    """Load 3D video frames and related metadata for optimizing consistency loss.
    File organization of the corresponding 3D video dataset should be
        color_down/frame_{__ID__:06d}.raw
        flow/flow_{__REF_ID__:06d}_{__TGT_ID__:06d}.raw
        mask/mask_{__REF_ID__:06d}_{__TGT_ID__:06d}.png
        metadata.npz: {'extrinsics': (N, 3, 4), 'intrinsics': (N, 4)}
        <flow_list.json>: [[i, j], ...]
    """

    def __init__(self, path: str, meta_file: str = None, params = None, suffix = ''):
        """
        Args:
            path: folder path of the 3D video
        """
        self.color_fmt = pjoin(path, "color_down", "frame_{:06d}.raw")
        if not os.path.isfile(self.color_fmt.format(0)):
            self.color_fmt = pjoin(path, "color_down", "frame_{:06d}.png")
        # self.color_fmt = pjoin(path, "color_flow", "frame_{:06d}.png")

        self.flow_fmt = pjoin(path, "flow", "flow_{:06d}_{:06d}.raw")
        self.reg_fmt = pjoin(path, "flow_reg", "reg_{:06d}_{:06d}.raw") # lbz added
        self.hba_fmt = pjoin(path, "flow_hba", "hba_{:06d}_{:06d}.npz") # lbz added

        self.gt_predicted_is_ready = params.gt_predicted_is_ready
        self.gt_prediction_prepro = params.gt_prediction_prepro
        self.preproID = params.preproID
        self.ablation_median_didabled = params.ablation_median_didabled
        if self.gt_predicted_is_ready:                
            if self.gt_prediction_prepro:
                if self.preproID == 0:
                    self.depth_gt_fmt = pjoin(path, "depth_gt_predicted{}_prepro0".format(suffix), "depth_{:06d}_gt_predicted_average.raw") # lbz added
                    self.mask_prepro_fmt = pjoin(path, "mask_rectified_average{}".format(suffix), "mask_{:06d}.png")
                if self.preproID == 1:
                    self.depth_gt_fmt = pjoin(path, "depth_gt_predicted{}_prepro1".format(suffix), "depth_{:06d}_gt_predicted_median.raw") # lbz added
                    self.mask_prepro_fmt = pjoin(path, "mask_rectified_median{}".format(suffix), "mask_{:06d}.png")
                    self.depth_projection_median_confidence_map_fmt = pjoin(path, "depth_median_confidence_map{}_ctm{}".format(suffix, params.confidence_tolerance_median), "frame_{:06d}.npz")
                if self.preproID == 2:
                    self.depth_gt_fmt = pjoin(path, "depth_gt_predicted{}_prepro2".format(suffix), "depth_{:06d}_gt_predicted_selection.raw") # lbz added
                    self.mask_prepro_fmt = pjoin(path, "mask_rectified_selection{}".format(suffix), "mask_{:06d}.png")
                    if params.upsampling_factor != 0:
                        self.depth_gt_fmt = pjoin(path, "depth_gt_predicted{}_prepro2_projection_pd{}_border0_smoothed".format(suffix, params.projection_distance), "depth_projection_frames_projection_median_{:06d}_smoothed.raw") # lbz added
                    else:
                        self.depth_gt_fmt = pjoin(path, "depth_gt_predicted{}_prepro2_projection_pd{}_border0".format(suffix, params.projection_distance), "depth_projection_frames_projection_median_{:06d}.raw") # lbz added
                    
                    if params.mask_final == 0: # median mask
                        self.mask_prepro_fmt = pjoin(path, "mask_rectified_selection{}".format(suffix), "mask_{:06d}.png")
                    if params.mask_final == 1: # projection mask
                        self.mask_prepro_fmt = pjoin(path, "mask_rectified_selection{}_projection_pd{}_border0".format(suffix, params.projection_distance), "mask_{:06d}.png")
                    
                    self.depth_projection_median_confidence_map_fmt = pjoin(path, "depth_projection_selection_confidence_map{}_ctp{}".format(suffix, params.confidence_tolerance_projection), "frame_{:06d}.npz")
                
                if self.preproID == 10:
                    if params.upsampling_factor != 0:
                        self.depth_gt_fmt = pjoin(path, "depth_gt_predicted{}_prepro1_projection_pd{}_border0_smoothed".format(suffix, params.projection_distance), "depth_projection_frames_projection_median_{:06d}_smoothed.raw") # lbz added
                    else:
                        self.depth_gt_fmt = pjoin(path, "depth_gt_predicted{}_prepro1_projection_pd{}_border0".format(suffix, params.projection_distance), "depth_projection_frames_projection_median_{:06d}.raw") # lbz added
                    
                    if params.mask_final == 0: # median mask
                        self.mask_prepro_fmt = pjoin(path, "mask_rectified_median{}".format(suffix), "mask_{:06d}.png")
                    if params.mask_final == 1: # projection mask
                        self.mask_prepro_fmt = pjoin(path, "mask_rectified_median{}_projection_pd{}_border0".format(suffix, params.projection_distance), "mask_{:06d}.png")
                    
                    self.depth_projection_median_confidence_map_fmt = pjoin(path, "depth_projection_median_confidence_map{}_ctm{}_ctp{}".format(suffix, params.confidence_tolerance_median, params.confidence_tolerance_projection), "frame_{:06d}.npz")
            else:
                self.depth_gt_fmt = pjoin(path, "depth_gt_predicted{}".format(suffix), "depth_{:06d}_gt_predicted_{:06d}_{:06d}.raw") # lbz added
            
            if self.ablation_median_didabled:
                self.depth_gt_fmt = pjoin(path, "depth_gt_predicted{}".format(suffix), "depth_{:06d}_gt_predicted_{:06d}_{:06d}.raw") # lbz added

            self.mask_fmt = pjoin(path, "mask_rectified{}".format(suffix), "mask_{:06d}_{:06d}.png")
        else:
            self.mask_fmt = pjoin(path, "mask", "mask_{:06d}_{:06d}.png")


        if meta_file is not None:
            with open(meta_file, "rb") as f:
                meta = np.load(f)
                self.extrinsics = torch.tensor(meta["extrinsics"], dtype=_dtype)
                self.intrinsics = torch.tensor(meta["intrinsics"], dtype=_dtype)
                # lbz20211220_scale
                # self.scale = torch.tensor(meta["scales"], dtype=_dtype)[:, 1].mean()
            assert (
                self.extrinsics.shape[0] == self.intrinsics.shape[0]
            ), "#extrinsics({}) != #intrinsics({})".format(
                self.extrinsics.shape[0], self.intrinsics.shape[0]
            )

        if self.gt_predicted_is_ready:
            if self.gt_prediction_prepro:
                if self.preproID == 0:
                    flow_list_fn = pjoin(path, "depth_gt_predicted_average_list{}.json".format(suffix))
                elif self.preproID == 1:
                    flow_list_fn = pjoin(path, "depth_gt_predicted_median_list{}.json".format(suffix))
                elif self.preproID == 2:
                    flow_list_fn = pjoin(path, "depth_gt_predicted_selection_list{}.json".format(suffix))
                elif self.preproID == 10:
                    flow_list_fn = pjoin(path, "depth_gt_predicted_median_list{}.json".format(suffix))
                else:
                    flow_list_fn = pjoin(path, "depth_gt_predicted_list{}.json".format(suffix))
            else:
                flow_list_fn = pjoin(path, "depth_gt_predicted_list{}.json".format(suffix))
            if self.ablation_median_didabled:
                flow_list_fn = pjoin(path, "depth_gt_predicted_median_list{}.json".format(suffix))
                gt_candidates_list_fn = pjoin(path, "depth_gt_predicted_list{}_amd{}.json".format(suffix, self.ablation_median_didabled))
        else:
            flow_list_fn = pjoin(path, "flow_list.json")
            
        print(flow_list_fn)

        if os.path.isfile(flow_list_fn):
            with open(flow_list_fn, "r") as f:
                self.flow_indices = json.load(f)
        else:
            names = os.listdir(os.path.dirname(self.flow_fmt))
            self.flow_indices = [
                self.parse_index_pair(name)
                for name in names
                if os.path.splitext(name)[-1] == os.path.splitext(self.flow_fmt)[-1]
            ]
            self.flow_indices = sampling.to_in_range(self.flow_indices)
        self.flow_indices = list(sampling.SamplePairs.to_one_way(self.flow_indices))

        if self.ablation_median_didabled:
            if os.path.isfile(gt_candidates_list_fn):
                with open(gt_candidates_list_fn, "r") as f:
                    self.gt_candidates_dict = json.load(f)
 
    def parse_index_pair(self, name):
        strs = os.path.splitext(name)[0].split("_")[-2:]
        return [int(s) for s in strs]

    def __getitem__(self, index: int):
        """Fetch tuples of data. index = i * (i-1) / 2 + j, where i > j for pair (i,j)
        So [-1+sqrt(1+8k)]/2 < i <= [1+sqrt(1+8k))]/2, where k=index. So
            i = floor([1+sqrt(1+8k))]/2)
            j = k - i * (i - 1) / 2.

        The number of image frames fetched, N, is not the 1, but computed
        based on what kind of consistency to be measured.
        For instance, geometry_consistency_loss requires random pairs as samples.
        So N = 2.
        If with more losses, say triplet one from temporal_consistency_loss. Then
            N = 2 + 3.

        Returns:
            stacked_images (N, C, H, W): image frames
            targets: {
                'extrinsics': torch.tensor (N, 3, 4), # extrinsics of each frame.
                                Each (3, 4) = [R, t].
                                    point_wolrd = R * point_cam + t
                'intrinsics': torch.tensor (N, 4), # (fx, fy, cx, cy) for each frame
                'geometry_consistency':
                    {
                        'indices':  torch.tensor (2),
                                    indices for corresponding pairs
                                        [(ref_index, tgt_index), ...]
                        'flows':    ((2, H, W),) * 2 in pixels.
                                    For k in range(2) (ref or tgt),
                                        pixel p = pixels[indices[b, k]][:, i, j]
                                    correspond to
                                        p + flows[k][b, :, i, j]
                                    in frame indices[b, (k + 1) % 2].
                        'masks':    ((1, H, W),) * 2. Masks of valid flow matches
                                    to compute the consistency in training.
                                    Values are 0 or 1.
                    }
            }

        """
        pair = self.flow_indices[index]

        indices = torch.tensor(pair)
        intrinsics = torch.stack([self.intrinsics[k] for k in pair], dim=0)
        extrinsics = torch.stack([self.extrinsics[k] for k in pair], dim=0)
        # intrinsics = torch.stack([self.intrinsics[k-1] for k in pair], dim=0) # trick for reception_room_003: k-1 rather than k
        # extrinsics = torch.stack([self.extrinsics[k-1] for k in pair], dim=0) # trick for reception_room_003: k-1 rather than k
        # lbz20211220_scale
        # scale = self.scale

        # print("len(self.intrinsics):", len(self.intrinsics))
        images = torch.stack(
            [load_color(self.color_fmt.format(k), channels_first=True) for k in pair],
            dim=0,
        )

        # lbz modified
        # images = torch.stack(
        #     [torch.from_numpy(cv2.imread(self.color_fmt.format(k))[:, :, ::-1] / 255).permute(2, 0, 1).contiguous().float() for k in pair],
        #     # [torch.from_numpy(cv2.resize(cv2.imread(self.color_fmt.format(k))[:, :, ::-1], (256, 384)) / 255).permute(2, 0, 1).contiguous().float() for k in pair],
        #     dim=0,
        # )
        flows = [
            load_flow(self.flow_fmt.format(k_ref, k_tgt), channels_first=True)
            for k_ref, k_tgt in [pair, pair[::-1]]
        ]
        masks = [
            load_mask(self.mask_fmt.format(k_ref, k_tgt), channels_first=True)
            for k_ref, k_tgt in [pair, pair[::-1]]
        ]
        if self.gt_predicted_is_ready:
            if self.gt_prediction_prepro:
                if not self.ablation_median_didabled:
                    masks_prepro = [
                        load_mask(self.mask_prepro_fmt.format(k_ref), channels_first=True)
                        for k_ref, _ in [pair, pair[::-1]]
                    ]
                    depth_projection_median_confidence_map = []
                    for k_ref, _ in [pair, pair[::-1]]:
                        with open(self.depth_projection_median_confidence_map_fmt.format(k_ref), "rb") as fp:
                            depth_projection_median_confidence_map.append(torch.tensor(np.load(fp)["points"])[None, ...])
                else:
                    cnt_pair = 0
                    for k_ref, _ in [pair, pair[::-1]]:
                        k_ref_candidates = self.gt_candidates_dict[str(k_ref)]
                        cnt = 0
                        for k_pair in k_ref_candidates:
                            mask = load_mask(self.mask_fmt.format(k_pair[0], k_pair[1]), channels_first=True) # (1, 288, 384)
                            depth = torch.from_numpy(load_raw_float32_image(self.depth_gt_fmt.format(k_ref, k_pair[0], k_pair[1]))).unsqueeze(0) # (1, 1, 288, 384)
                            if cnt == 0:
                                mask_stack = mask
                                depth_stack = depth
                                cnt += 1
                            else:
                                mask_stack = torch.cat((mask_stack, mask), dim=0)
                                depth_stack = torch.cat((depth_stack, depth), dim=1)
                            
                        if cnt_pair == 0:
                            masks_candidates = [mask_stack]
                            depths_candidates = [depth_stack]
                            cnt_pair += 1
                        else:
                            masks_candidates.append(mask_stack)
                            depths_candidates.append(depth_stack)     
        # regs = torch.stack(
        #     [torch.from_numpy(load_raw_float32_image(self.reg_fmt.format(k_ref, k_tgt))).permute(2, 0, 1)
        #     for k_ref, k_tgt in [pair, pair[::-1]]],
        #     dim=0,
        # )
        # hbas = [
        #     np.load(self.hba_fmt.format(k_ref, k_tgt))
        #     for k_ref, k_tgt in [pair, pair[::-1]]
        # ]
        if self.gt_predicted_is_ready:
            # print(self.depth_gt_fmt.format(pair[0], pair[0], pair[1]))
            if self.gt_prediction_prepro:
                if not self.ablation_median_didabled:
                    gt_predictions = [torch.from_numpy(load_raw_float32_image(self.depth_gt_fmt.format(pair[0]))).unsqueeze(0), \
                        torch.from_numpy(load_raw_float32_image(self.depth_gt_fmt.format(pair[1]))).unsqueeze(0)]
                    # print(gt_predictions[0].shape)
                    metadata = {
                        "extrinsics": extrinsics,
                        "intrinsics": intrinsics,
                        # lbz20211220_scale
                        # "scale": scale,
                        "geometry_consistency": {
                            "indices": indices,
                            "flows": flows,
                            "masks": masks,
                            # "regs": regs,
                            # "hbas": hbas,
                            "gt_predictions": gt_predictions,
                            "masks_prepro": masks_prepro,
                            "depth_projection_median_confidence_map": depth_projection_median_confidence_map,
                        },
                    }
                else:
                    metadata = {
                        "extrinsics": extrinsics,
                        "intrinsics": intrinsics,
                        # lbz20211220_scale
                        # "scale": scale,
                        "geometry_consistency": {
                            "indices": indices,
                            "flows": flows,
                            "masks": masks,
                            # "regs": regs,
                            # "hbas": hbas,
                            "gt_predictions": depths_candidates,
                            "masks_prepro": masks_candidates,
                            "depth_projection_median_confidence_map": masks_candidates, # no use for ablation_median_didabled case
                        },
                    }
            else:
                gt_predictions = [torch.from_numpy(load_raw_float32_image(self.depth_gt_fmt.format(pair[0], pair[0], pair[1]))).unsqueeze(0), \
                    torch.from_numpy(load_raw_float32_image(self.depth_gt_fmt.format(pair[1], pair[0], pair[1]))).unsqueeze(0)]
            
                metadata = {
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    # lbz20211220_scale
                    # "scale": scale,
                    "geometry_consistency": {
                        "indices": indices,
                        "flows": flows,
                        "masks": masks,
                        # "regs": regs,
                        # "hbas": hbas,
                        "gt_predictions": gt_predictions,
                    },
                }
        else:
            metadata = {
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                # lbz20211220_scale
                # "scale": scale,
                "geometry_consistency": {
                    "indices": indices,
                    "flows": flows,
                    "masks": masks,
                    # "regs": regs,
                    # "hbas": hbas,
                },
            }

        if getattr(self, "scales", None):
            if isinstance(self.scales, dict):
                metadata["scales"] = torch.stack(
                    [torch.Tensor([self.scales[k]]) for k in pair], dim=0
                )
            else:
                metadata["scales"] = torch.Tensor(
                    [self.scales, self.scales]).reshape(2, 1)

        return (images, metadata)

    def __len__(self):
        return len(self.flow_indices)


class VideoFrameDataset(data.Dataset):
    """Load video frames from
        color_fmt.format(frame_id)
    """

    def __init__(self, color_fmt, frames=None):
        """
        Args:
            color_fmt: e.g., <video_dir>/frame_{:06d}.raw
        """
        self.color_fmt = color_fmt

        if frames is None:
            files = os.listdir(os.path.dirname(self.color_fmt))
            self.frames = range(len(files))
        else:
            self.frames = frames

    def __getitem__(self, index):
        """Fetch image frame.
        Returns:
            image (C, H, W): image frames
        """
        frame_id = self.frames[index]
        image = load_color(self.color_fmt.format(frame_id), channels_first=True)
        meta = {"frame_id": frame_id}
        return image, meta

    def __len__(self):
        return len(self.frames)
