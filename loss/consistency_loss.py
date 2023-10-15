#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_helpers import _device
from utils.geometry import (
    pixel_grid,
    focal_length,
    project,
    pixels_to_points,
    reproject_points,
    reproject_points2,
    to_worldspace,
    to_camera,
    ends_common_perpendicular,
    gt_cvd,
    gt_spatial_loss_method,
    sample,
    pixels_to_homogeneous_coord,
    pixels_to_rays,

)

import pdb
from utils.image_io import save_raw_float32_image
import os

def select_tensors(x):
    """
    x (B, N, C, H, W) -> (N, B, C, H, W)
    Each batch (B) is composed of a pair or more samples (N).
    """
    return x.transpose(0, 1)


def weighted_mse_loss(input, target, weights, dim=1, eps=1e-6):
    """
        Args:
            input (B, C, H, W)
            target (B, C, H, W)
            weights (B, 1, H, W)

        Returns:
            scalar
    """
    assert (
        input.ndimension() == target.ndimension()
        and input.ndimension() == weights.ndimension()
    )
    # normalize to sum=1
    B = weights.shape[0]
    weights_sum = torch.sum(weights.view(B, -1), dim=-1).view(B, 1, 1, 1)
    weights_sum = torch.clamp(weights_sum, min=eps)
    weights_n = weights / weights_sum

    sq_error = torch.sum((input - target) ** 2, dim=dim, keepdim=True)  # BHW
    return torch.sum((weights_n * sq_error).reshape(B, -1), dim=1)


def weighted_rmse_loss(input, target, weights, dim=1, eps=1e-6):
    """
        Args:
            input (B, C, H, W)
            target (B, C, H, W)
            weights (B, 1, H, W)

        Returns:
            scalar = weighted_mean(rmse_along_dim)
    """
    assert (
        input.ndimension() == target.ndimension()
        and input.ndimension() == weights.ndimension()
    )
    # normalize to sum=1
    B = weights.shape[0]
    weights_sum = torch.sum(weights.view(B, -1), dim=-1).view(B, 1, 1, 1)
    weights_sum = torch.clamp(weights_sum, min=eps)
    weights_n = weights / weights_sum

    diff = torch.norm(input - target, dim=dim, keepdim=True)
    return torch.sum((weights_n * diff).reshape(B, -1), dim=1)


def weighted_mean_loss(x, weights, eps=1e-6):
    """
        Args:
            x (B, ...)
            weights (B, ...)

        Returns:
            a scalar
    """
    assert x.ndimension() == weights.ndimension() and x.shape[0] == weights.shape[0]
    # normalize to sum=1
    B = weights.shape[0]
    weights_sum = torch.sum(weights.view(B, -1), dim=-1).view(B, 1, 1, 1)
    weights_sum = torch.clamp(weights_sum, min=eps)
    weights_n = weights / weights_sum

    return torch.sum((weights_n * x).reshape(B, -1), dim=1)


class ConsistencyLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.dist = torch.abs

    def geometry_consistency_loss(self, points_cam, metadata, pixels):
        """Geometry Consistency Loss.

        For each pair as specified by indices,
            geom_consistency = reprojection_error + disparity_error
        reprojection_error is measured in the screen space of each camera in the pair.

        Args:
            points_cam (B, N, 3, H, W): points in local camera coordinate.
            pixels (B, N, 2, H, W)
            metadata: dictionary of related metadata to compute the loss. Here assumes
                metadata include entries as below.
                {
                    'extrinsics': torch.tensor (B, N, 3, 4), # extrinsics of each frame.
                                    Each (3, 4) = [R, t]
                    'intrinsics': torch.tensor (B, N, 4), # (fx, fy, cx, cy)
                    'geometry_consistency':
                        {
                            'flows':    (B, 2, H, W),) * 2 in pixels.
                                        For k in range(2) (ref or tgt),
                                            pixel p = pixels[indices[b, k]][:, i, j]
                                        correspond to
                                            p + flows[k][b, :, i, j]
                                        in frame indices[b, (k + 1) % 2].
                            'masks':    ((B, 1, H, W),) * 2. Masks of valid flow
                                        matches. Values are 0 or 1.
                        }
                }
        """
        geom_meta = metadata["geometry_consistency"]
        points_cam_pair = select_tensors(points_cam)
        extrinsics = metadata["extrinsics"]
        extrinsics_pair = select_tensors(extrinsics)
        intrinsics = metadata["intrinsics"]
        intrinsics_pair = select_tensors(intrinsics)
        pixels_pair = select_tensors(pixels)

        flows_pair = (flows for flows in geom_meta["flows"])
        masks_pair = (masks for masks in geom_meta["masks"])

        reproj_losses, disp_losses, depth_losses, g1_g2_losses, epipolar_losses = [], [], [], [], []
        inv_idxs = [1, 0]
        cnt = 0
        G1s, G1_diff_list, G1_diff_max_list, masks = [], [], [], []
        G1s_shift, G1_shift_diff_list, G1_shift_diff_max_list, masks_shift = [], [], [], []
        ending_tgt_list = []

        if not self.opt.gt_predicted_is_ready:
            for (
                points_cam_ref,
                tgt_points_cam_tgt,
                pixels_ref,
                flows_ref,
                masks_ref,
                intrinsics_ref,
                intrinsics_tgt,
                extrinsics_ref,
                extrinsics_tgt,
            ) in zip(
                points_cam_pair,
                points_cam_pair[inv_idxs],
                pixels_pair,
                flows_pair,
                masks_pair,
                intrinsics_pair,
                intrinsics_pair[inv_idxs],
                extrinsics_pair,
                extrinsics_pair[inv_idxs],
            ):
                G1_list = []
                for flow in [flows_ref + 1, flows_ref]:
                    # the pair of [flows_ref, flows_ref+1] is used to check is G1 is reliable
                    # if ||G1(flows_ref) - G1(flows_ref+1)|| is small, like less than 1, when the flow shifts 1 pixel
                    # the G1 is reliable.

                    # change to camera space for target_camera
                    # points_cam_tgt = reproject_points(points_cam_ref, extrinsics_ref, extrinsics_tgt)
                    points_cam_tgt, pos_cam_ref_tgt = reproject_points2(points_cam_ref, extrinsics_ref, extrinsics_tgt)
                    matched_pixels_tgt = pixels_ref + flow# flows_ref # flow-warped
                    pixels_tgt = project(points_cam_tgt, intrinsics_tgt) # depth-warped
                    
                    if self.opt.lambda_reprojection > 0:
                        reproj_dist = torch.norm(pixels_tgt - matched_pixels_tgt,
                            dim=1, keepdim=True)
                        reproj_losses.append(
                            weighted_mean_loss(self.dist(reproj_dist), masks_ref)
                        )

                    if self.opt.lambda_view_baseline > 0:
                        # disparity consistency
                        f = torch.mean(focal_length(intrinsics_ref))
                        # warp points in target image grid target camera coordinates to
                        # reference image grid
                        warped_tgt_points_cam_tgt = sample(
                            tgt_points_cam_tgt, matched_pixels_tgt
                        )

                        disp_diff = 1.0 / points_cam_tgt[:, -1:, ...] \
                            - 1.0 / warped_tgt_points_cam_tgt[:, -1:, ...]
                        
                        disp_losses.append(
                            f * weighted_mean_loss(self.dist(disp_diff), masks_ref)
                        )
                            
                        if self.opt.straight_line_method:
                            # pdb.set_trace()

                            # points_image_plane_cam_ref (p) and warped_tgt_points_cam_tgt (q) are flow correspondences in image planes
                            # if flow (matched_pixels_tgt) is small, warped_tgt_points_cam_tgt and tgt_points_cam_tgt would be very similar.
                            points_world_warped_tgt, cam_world_tgt = to_worldspace(warped_tgt_points_cam_tgt, extrinsics_tgt)
                            points_image_plane_cam_ref = points_cam_ref / -points_cam_ref[:, 2:, :, :]
                            points_world_ref, cam_world_ref = to_worldspace(points_cam_ref, extrinsics_ref) # x_j(p) in world space
                            points_image_plane_world_ref, _ = to_worldspace(points_image_plane_cam_ref, extrinsics_ref) # p in world space

                            # the center of the reference image plane
                            plane_center_ref = torch.tensor([0, 0, -1]).reshape(1, 3, 1, 1).cuda() * torch.ones_like(points_cam_ref)
                            plane_center_world_ref, _ = to_worldspace(plane_center_ref, extrinsics_ref)

                            # project x's to the image plane of reference camera
                            # the projected points are used as epipoles, which is too small to be good precision.
                            points_ref_warped_tgt = to_camera(points_world_warped_tgt, extrinsics_ref) # projecting q to reference camera space
                            points_image_plane_ref_warped_tgt = points_ref_warped_tgt / -points_ref_warped_tgt[:, 2:, :, :] # prejected q in image plane of reference camera
                            points_world_ref_warped_tgt, _ = to_worldspace(points_image_plane_ref_warped_tgt, extrinsics_ref) # projected q in world space
                            
                            ending_ref, ending_tgt, mask_depth = gt_spatial_loss_method(cam_world_ref, points_world_ref, points_image_plane_world_ref, plane_center_world_ref, points_world_ref_warped_tgt, cam_world_tgt, points_world_warped_tgt, masks_ref, self.opt.eps, self.opt.eps2, self.opt.depth_mean_min, self.opt.max_depth, self.opt.pos_dist_min, self.opt.parallel_lines_out)

                            if 0: # distance
                                G1_pre = torch.norm(ending_ref - cam_world_ref, dim=1, keepdim=True)
                                G2_pre = torch.norm(ending_tgt - cam_world_tgt, dim=1, keepdim=True)
                                D1_pre = torch.norm(points_world_ref - cam_world_ref, dim=1, keepdim=True)
                                D2_pre = torch.norm(points_world_warped_tgt - cam_world_tgt, dim=1, keepdim=True)
                            else: # real depth
                                ending_ref_cam = to_camera(ending_ref, extrinsics_ref)
                                ending_tgt_cam = to_camera(ending_tgt, extrinsics_tgt)
                                # pdb.set_trace()
                                G1_pre = torch.abs(ending_ref_cam[:, 2:, ...])
                                G2_pre = torch.abs(ending_tgt_cam[:, 2:, ...])
                                D1_pre = torch.norm(points_world_ref - cam_world_ref, dim=1, keepdim=True)
                                D2_pre = torch.norm(points_world_warped_tgt - cam_world_tgt, dim=1, keepdim=True)

                            G1, D1, G2, D2 = G1_pre.clone(), D1_pre.clone(), G2_pre.clone(), D2_pre.clone()
                            if torch.isinf(G2.max()):
                                pdb.set_trace()
                            G1[G1_pre <= self.opt.eps] = self.opt.eps
                            D1[D1_pre <= self.opt.eps] = self.opt.eps
                            G2[G2_pre <= self.opt.eps] = self.opt.eps
                            D2[D2_pre <= self.opt.eps] = self.opt.eps
                            
                            ref_tgt_diff = torch.norm(points_world_ref - points_world_warped_tgt, dim=1, keepdim=True)
                            g1_g2_diff = torch.norm(ending_ref - ending_tgt, dim=1, keepdim=True)

                            G1_list.append(G2)
                            # G1_list.append(G1)

                if self.opt.straight_line_method:
                    mask_ratio = torch.sum(masks_ref, (1, 2, 3)) / (masks_ref.shape[2] * masks_ref.shape[3])
                    mask_ratio[mask_ratio >= 0.6] = 1 # small camera motion
                    mask_ratio[mask_ratio <  0.6] = 0 # large camera motion
                    mask_ratio[mask_ratio == 1] = 5
                    mask_ratio[mask_ratio == 0] = 0.1
                    ref_tgt_depth_diff_weight = mask_ratio.clone().view(-1, 1, 1, 1)
                    
                    G1_diff_ratio = self.dist(G1_list[0] / G1_list[1])
                    ratio = G1_diff_ratio.clone()
                    threshold = self.opt.g1_diff_threshold
                    ratio[torch.abs(ratio - 1) >= threshold] = 2.0 # bad
                    ratio[torch.abs(ratio - 1) <  threshold] = 0.0 # good
                    ratio_bad = ratio / 2
                    mask_sen = 1 - ratio_bad

                    if self.opt.loss_case == 1:
                        depth_losses_map = self.opt.pred_ref_gt_diff_weight * self.dist(torch.log10(1 + G1) - torch.log10(1 + D1)) + \
                                self.opt.pred_tgt_gt_diff_weight * self.dist(torch.log10(1 + G2) - torch.log10(1 + D2)) + \
                                self.opt.ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff)
                                # ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff)

                        depth_losses.append(
                            weighted_mean_loss(depth_losses_map, (masks_ref * mask_depth * mask_sen).float())
                        )
                    elif self.opt.loss_case == 0:
                        depth_losses_map = self.opt.pred_ref_gt_diff_weight * self.dist(torch.log10(G1) - torch.log10(D1)) + \
                                self.opt.pred_tgt_gt_diff_weight * self.dist(torch.log10(G2) - torch.log10(D2)) + \
                                self.opt.ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff)

                        depth_losses.append(
                            weighted_mean_loss(depth_losses_map, (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 2:
                        depth_losses.append(
                            weighted_mean_loss(self.dist(ref_tgt_diff) / (self.dist(g1_g2_diff) + 1e-1), (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 3:
                        depth_losses.append(
                            weighted_mean_loss( \
                                self.opt.pred_ref_gt_diff_weight * self.dist(torch.log10(G1) - torch.log10(D1)) + \
                                self.opt.pred_tgt_gt_diff_weight * self.dist(torch.log10(G2) - torch.log10(D2)) + \
                                self.dist(ref_tgt_diff) / (self.dist(g1_g2_diff) + 0.01)
                                , (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 4:
                        ## Let's say the unit vector defining the direction of the two lines are v1 and v2. 
                        # Also consider the vector connecting the two points in three to be D1D2. The loss is:
                        # ||D1D2 * v1|| + ||D1D2 * v2||
                        # where * is dot product.
                        vec1 = F.normalize(points_world_ref - cam_world_ref, dim=1, p=2)
                        vec2 = F.normalize(points_world_warped_tgt - cam_world_tgt, dim=1, p=2)
                        ref_tgt = points_world_ref - points_world_warped_tgt
                        ref_tgt_vec1_dot = torch.sum(ref_tgt * vec1, dim=1, keepdim=True)
                        ref_tgt_vec2_dot = torch.sum(ref_tgt * vec2, dim=1, keepdim=True)
                        # pdb.set_trace()
                        depth_losses.append(
                            weighted_mean_loss(self.dist(ref_tgt_vec1_dot) + self.dist(ref_tgt_vec2_dot), (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 5:
                        ## Let's say the unit vector defining the direction of the two lines are v1 and v2. 
                        # Also consider the vector connecting the two points in three to be D1D2. The loss is:
                        # (||D1D2 * v1|| + ||D1D2 * v2||) / (1 - (v1 * v2)^2)
                        # where * is dot product.
                        vec1 = F.normalize(points_world_ref - cam_world_ref, dim=1, p=2)
                        vec2 = F.normalize(points_world_warped_tgt - cam_world_tgt, dim=1, p=2)
                        ref_tgt = points_world_ref - points_world_warped_tgt
                        ref_tgt_vec1_dot = torch.sum(ref_tgt * vec1, dim=1, keepdim=True)
                        ref_tgt_vec2_dot = torch.sum(ref_tgt * vec2, dim=1, keepdim=True)
                        vec1_vec2_dot = torch.sum(vec1 * vec2, dim=1, keepdim=True)
                        # pdb.set_trace()
                        depth_losses.append(
                            weighted_mean_loss((self.dist(ref_tgt_vec1_dot) + self.dist(ref_tgt_vec2_dot)) / (1 - (vec1_vec2_dot ** 2 - 1e-2)), (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 6:
                        ## Let's say the unit vector defining the direction of the two lines are v1 and v2. 
                        # Also consider the vector connecting the two points in three to be D1D2 (normalized). The loss is:
                        # |(v1 x v2) x D1D2|
                        # where x is cross product.
                        vec1 = F.normalize(points_world_ref - cam_world_ref, dim=1, p=2)
                        vec2 = F.normalize(points_world_warped_tgt - cam_world_tgt, dim=1, p=2)
                        ref_tgt = F.normalize(points_world_ref - points_world_warped_tgt, dim=1, p=2)
                        vec1_vec2_ref_tgt_cross = torch.cross(torch.cross(vec1, vec2, dim=1), ref_tgt, dim=1)
                        # pdb.set_trace()
                        depth_losses.append(
                            weighted_mean_loss(torch.norm(vec1_vec2_ref_tgt_cross, dim=1, keepdim=True), (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 7:
                        ## Let's say the unit vector defining the direction of the two lines are v1 and v2. 
                        # Also consider the vector connecting the two points in three to be D1D2 (normalized). The loss is:
                        # |(v1 x v2) x D1D2| + |log(G1) - log(D1)| + |log(G2) - log(D2)|
                        # where x is cross product.
                        vec1 = F.normalize(points_world_ref - cam_world_ref, dim=1, p=2)
                        vec2 = F.normalize(points_world_warped_tgt - cam_world_tgt, dim=1, p=2)
                        ref_tgt = F.normalize(points_world_ref - points_world_warped_tgt, dim=1, p=2)
                        vec1_vec2_ref_tgt_cross = torch.cross(torch.cross(vec1, vec2, dim=1), ref_tgt, dim=1)
                        # pdb.set_trace()
                        depth_losses.append(
                            weighted_mean_loss( \
                                self.opt.pred_ref_gt_diff_weight * self.dist(torch.log10(G1) - torch.log10(D1)) + \
                                self.opt.pred_tgt_gt_diff_weight * self.dist(torch.log10(G2) - torch.log10(D2)) + \
                                torch.norm(vec1_vec2_ref_tgt_cross, dim=1, keepdim=True), (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 8:
                        ## Let's say the unit vector defining the direction of the two lines are v1 and v2. 
                        # Also consider the vector connecting the two points in three to be D1D2 (normalized). The loss is:
                        # (|D1D2| / (|G1G2 + 1e-2|)) + |(v1 x v2) x D1D2|
                        # where x is cross product.
                        vec1 = F.normalize(points_world_ref - cam_world_ref, dim=1, p=2)
                        vec2 = F.normalize(points_world_warped_tgt - cam_world_tgt, dim=1, p=2)
                        ref_tgt = F.normalize(points_world_ref - points_world_warped_tgt, dim=1, p=2)
                        vec1_vec2_ref_tgt_cross = torch.cross(torch.cross(vec1, vec2, dim=1), ref_tgt, dim=1)
                        # pdb.set_trace()
                        depth_losses.append(
                            weighted_mean_loss( \
                                self.dist(ref_tgt_diff) / (self.dist(g1_g2_diff) + 1e-2) + \
                                torch.norm(vec1_vec2_ref_tgt_cross, dim=1, keepdim=True), (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 9:
                        ## Let's say the unit vector defining the direction of the two lines are v1 and v2. 
                        # Also consider the vector connecting the two points in three to be D1D2. The loss is:
                        # (||D1D2|| / (1 - (v1 * v2)^2)
                        # where * is dot product.
                        vec1 = F.normalize(points_world_ref - cam_world_ref, dim=1, p=2)
                        vec2 = F.normalize(points_world_warped_tgt - cam_world_tgt, dim=1, p=2)
                        ref_tgt = points_world_ref - points_world_warped_tgt
                        vec1_vec2_dot = torch.sum(vec1 * vec2, dim=1, keepdim=True)
                        # pdb.set_trace()
                        depth_losses.append(
                            weighted_mean_loss((self.dist(ref_tgt_diff)) / (1 - (vec1_vec2_dot ** 2 - 1e-2)), (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 10:
                        # (|log(G1) - log(D1)| + |log(G2) - log(D2)|) + （|D1D2| / (|G1G2 + 1e-1|))
                        # pdb.set_trace()
                        depth_losses.append(
                            weighted_mean_loss( \
                                self.opt.pred_ref_gt_diff_weight * self.dist(torch.log10(G1) - torch.log10(D1)) + \
                                self.opt.pred_tgt_gt_diff_weight * self.dist(torch.log10(G2) - torch.log10(D2)) + \
                                self.opt.ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff) / (self.dist(g1_g2_diff) + 1e-1), \
                                    (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 11:
                        ## Let's say the unit vector defining the direction of the two lines are v1 and v2. 
                        # Also consider the vector connecting the two points in three to be D1D2. The loss is:
                        # (|D1D2| / ((1 - (v1 * v2)^2 + 1e-2) * (|G1G2 + 1e-2|)))
                        # where * is dot product.
                        vec1 = F.normalize(points_world_ref - cam_world_ref, dim=1, p=2)
                        vec2 = F.normalize(points_world_warped_tgt - cam_world_tgt, dim=1, p=2)
                        ref_tgt = points_world_ref - points_world_warped_tgt
                        vec1_vec2_dot = torch.sum(vec1 * vec2, dim=1, keepdim=True)
                        # pdb.set_trace()
                        depth_losses.append(
                            weighted_mean_loss((self.dist(ref_tgt_diff)) / ((1 - (vec1_vec2_dot ** 2) + 1e-1) * (self.dist(g1_g2_diff) + 1e-1)), (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 12:
                        ## Let's say the unit vector defining the direction of the two lines are v1 and v2. 
                        # Also consider the vector connecting the two points in three to be D1D2. The loss is:
                        # (|log(G1) - log(D1)| + |log(G2) - log(D2)|) + (|D1D2| / ((1 - (v1 * v2)^2 + 1e-2) * (|G1G2 + 1e-2|)))
                        # where * is dot product.
                        vec1 = F.normalize(points_world_ref - cam_world_ref, dim=1, p=2)
                        vec2 = F.normalize(points_world_warped_tgt - cam_world_tgt, dim=1, p=2)
                        ref_tgt = points_world_ref - points_world_warped_tgt
                        vec1_vec2_dot = torch.sum(vec1 * vec2, dim=1, keepdim=True)
                        # pdb.set_trace()
                        depth_losses.append(
                            weighted_mean_loss( \
                                self.opt.pred_ref_gt_diff_weight * self.dist(torch.log10(G1) - torch.log10(D1)) + \
                                self.opt.pred_tgt_gt_diff_weight * self.dist(torch.log10(G2) - torch.log10(D2)) + \
                                self.opt.ref_tgt_depth_diff_weight * (self.dist(ref_tgt_diff) / ((1 - (vec1_vec2_dot ** 2) + 1e-1) * (self.dist(g1_g2_diff) + 1e-1))) \
                                , (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 13:
                        D1_proj = project(to_camera(points_world_ref, extrinsics_tgt), intrinsics_tgt)
                        G1_proj = project(to_camera(ending_ref, extrinsics_tgt), intrinsics_tgt)
                        D2_proj = project(to_camera(points_world_warped_tgt, extrinsics_ref), intrinsics_ref)
                        G2_proj = project(to_camera(ending_tgt, extrinsics_ref), intrinsics_ref)
                        depth_losses_map = self.opt.pred_ref_gt_diff_weight * torch.norm(D1_proj - G1_proj, dim=1, keepdim=True) + \
                            self.opt.pred_tgt_gt_diff_weight * torch.norm(D2_proj - G2_proj, dim=1, keepdim=True) + \
                            self.opt.ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff)
                            # ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff)
                            
                        depth_losses.append(
                            weighted_mean_loss(depth_losses_map, (masks_ref * mask_depth * mask_sen).float())
                        )
                    elif self.opt.loss_case == 14:
                        D1_proj2 = project(to_camera(points_world_ref, extrinsics_tgt), intrinsics_tgt)
                        G1_proj2 = project(to_camera(ending_ref, extrinsics_tgt), intrinsics_tgt)
                        D2_proj1 = project(to_camera(points_world_warped_tgt, extrinsics_ref), intrinsics_ref)
                        G2_proj1 = project(to_camera(ending_tgt, extrinsics_ref), intrinsics_ref)
                        D1_proj1 = project(to_camera(points_world_ref, extrinsics_ref), intrinsics_ref)
                        D2_proj2 = project(to_camera(points_world_warped_tgt, extrinsics_tgt), intrinsics_tgt)

                        depth_losses_map = self.opt.pred_ref_gt_diff_weight * torch.norm(D1_proj2 - G1_proj2, dim=1, keepdim=True) + \
                            self.opt.pred_tgt_gt_diff_weight * torch.norm(D2_proj1 - G2_proj1, dim=1, keepdim=True) + \
                            self.opt.ref_tgt_depth_diff_weight * (torch.norm(D1_proj2 - D2_proj2, dim=1, keepdim=True) + \
                                torch.norm(D1_proj1 - D2_proj1, dim=1, keepdim=True))
                            # ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff)
                        pdb.set_trace()
                        depth_losses.append(
                            weighted_mean_loss(depth_losses_map, (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 15:
                        D1_proj2 = project(to_camera(points_world_ref, extrinsics_tgt), intrinsics_tgt)
                        G1_proj2 = project(to_camera(ending_ref, extrinsics_tgt), intrinsics_tgt)
                        D2_proj1 = project(to_camera(points_world_warped_tgt, extrinsics_ref), intrinsics_ref)
                        G2_proj1 = project(to_camera(ending_tgt, extrinsics_ref), intrinsics_ref)
                        D1_proj1 = project(to_camera(points_world_ref, extrinsics_ref), intrinsics_ref)
                        D2_proj2 = project(to_camera(points_world_warped_tgt, extrinsics_tgt), intrinsics_tgt)

                        depth_losses_map = torch.norm(D1_proj2 - G1_proj2, dim=1, keepdim=True) + \
                            torch.norm(D2_proj1 - G2_proj1, dim=1, keepdim=True) + \
                            torch.norm(D1_proj2 - D2_proj2, dim=1, keepdim=True) + \
                            torch.norm(D1_proj1 - D2_proj1, dim=1, keepdim=True) + \
                            self.opt.ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff)

                        # pdb.set_trace()

                        depth_losses.append(
                            weighted_mean_loss(depth_losses_map, (masks_ref * mask_depth).float())
                        )
                    elif self.opt.loss_case == 16:
                        G1_post = G1 - (((G1 >= D1).float() * 2 - 1) * g1_g2_diff * 100)
                        G2_post = G2 - (((G2 >= D2).float() * 2 - 1) * g1_g2_diff * 100)
                        depth_losses_map = self.opt.pred_ref_gt_diff_weight * self.dist(torch.log10(1 + G1_post) - torch.log10(1 + D1)) + \
                                self.opt.pred_tgt_gt_diff_weight * self.dist(torch.log10(1 + G2_post) - torch.log10(1 + D2)) + \
                                self.opt.ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff)
                                # ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff)

                        depth_losses.append(
                            weighted_mean_loss(depth_losses_map, (masks_ref * mask_depth * mask_sen).float())
                        )
                    elif self.opt.loss_case == 17:
                        vec1 = F.normalize(points_world_ref - cam_world_ref, dim=1, p=2)
                        vec2 = F.normalize(points_world_warped_tgt - cam_world_tgt, dim=1, p=2)
                        ref_tgt = points_world_ref - points_world_warped_tgt
                        vec1_vec2_dot = torch.sum(vec1 * vec2, dim=1, keepdim=True)
                        sin_theta = (1 - (vec1_vec2_dot ** 2) + 1e-6)
                        # pdb.set_trace()
                        depth_losses.append(
                            weighted_mean_loss((self.dist(ref_tgt_diff)) / ((1 - (vec1_vec2_dot ** 2) + 1e-6) * (self.dist(g1_g2_diff) + 1e-1)), (masks_ref * mask_depth).float())
                        )
                        depth_losses_map = self.opt.pred_ref_gt_diff_weight * self.dist(torch.log10(1 + G1) - torch.log10(1 + D1)) + \
                                self.opt.pred_tgt_gt_diff_weight * self.dist(torch.log10(1 + G2) - torch.log10(1 + D2)) + \
                                self.opt.ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff) + \
                                100 * self.dist(sin_theta)
                                
                        depth_losses.append(
                            weighted_mean_loss(depth_losses_map, (masks_ref * mask_depth * mask_sen).float())
                        )
                    elif self.opt.loss_case == 18:
                        depth_losses_map = self.opt.pred_ref_gt_diff_weight * self.dist(torch.log10(1 + G2) - torch.log10(1 + D2)) + \
                                self.opt.ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff)
                                
                        depth_losses.append(
                            weighted_mean_loss(depth_losses_map, (masks_ref * mask_depth * mask_sen).float())
                        )
                    elif self.opt.loss_case == 19:
                        D1 = torch.norm(points_world_ref - cam_world_ref, dim=1, keepdim=True)                        
                        depth_losses_map = self.opt.pred_ref_gt_diff_weight * self.dist(torch.log10(1 + G2) - torch.log10(1 + D2)) + \
                                self.opt.ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff / (D1 * D2))
                                
                        depth_losses.append(
                            weighted_mean_loss(depth_losses_map, (masks_ref).float())
                        )
                    elif self.opt.loss_case == 20:
                        D2_proj = project(to_camera(points_world_warped_tgt, extrinsics_ref), intrinsics_ref)
                        G2_proj = project(to_camera(ending_tgt, extrinsics_ref), intrinsics_ref)
                        # depth_losses_map = self.opt.pred_ref_gt_diff_weight * self.dist(torch.log10(1 + G2_proj) - torch.log10(1 + D2_proj)) + \
                        depth_losses_map = self.opt.pred_tgt_gt_diff_weight * torch.norm(D2_proj - G2_proj, dim=1, keepdim=True) + \
                                self.opt.ref_tgt_depth_diff_weight * self.dist(ref_tgt_diff)
                                
                        depth_losses.append(
                            weighted_mean_loss(depth_losses_map, (masks_ref * mask_depth * mask_sen).float())
                        )
                    else:
                        pass
                    
                if (self.opt.test_mode or not self.opt.gt_predicted_is_ready) and self.opt.straight_line_method:
                    G1s.append(G2)
                    G1s_shift.append(G1_list[0])
                    masks.append((masks_ref * mask_depth * mask_sen).float()[0].permute(1, 2, 0))
                    ending_tgt_list.append(ending_tgt)

            if (self.opt.test_mode or not self.opt.gt_predicted_is_ready) and self.opt.straight_line_method:
                # pdb.set_trace()
                import numpy as np
                index1 = metadata["geometry_consistency"]["indices"][0][0].item()
                index2 = metadata["geometry_consistency"]["indices"][0][1].item()
                
                points_cloud_dir_1 = self.opt.path + 'points_cloud/frame_{}.npz'.format(str(index2).zfill(6))
                points_cloud_dir_2 = self.opt.path + 'points_cloud/frame_{}.npz'.format(str(index1).zfill(6))
                
                if not os.path.isfile(points_cloud_dir_1):
                    np.savez(points_cloud_dir_1, points=ending_tgt_list[0].cpu().numpy())
                if not os.path.isfile(points_cloud_dir_2):
                    np.savez(points_cloud_dir_2, points=ending_tgt_list[1].cpu().numpy())

                if self.opt.straight_line_method:
                    # pdb.set_trace()
                    import cv2
                    from utils import visualization
                    G10_np_shift = G1s_shift[0].cpu().detach().numpy()[0, 0, ...]
                    G11_np_shift = G1s_shift[1].cpu().detach().numpy()[0, 0, ...]
                    G10_np = G1s[0].cpu().detach().numpy()[0, 0, ...]
                    G11_np = G1s[1].cpu().detach().numpy()[0, 0, ...]
                    inv_depth0_shift = 1.0 / np.clip(G10_np_shift, 0.1, 100)
                    inv_depth1_shift = 1.0 / np.clip(G11_np_shift, 0.1, 100)
                    inv_depth0 = 1.0 / G10_np
                    inv_depth1 = 1.0 / G11_np
                    vis_depth_scale = max(inv_depth0.max(), inv_depth1.max())

                    mask0 = masks[0].cpu().numpy() 
                    mask1 = masks[1].cpu().numpy() 
                    inv_depth_vis0_shift = visualization.visualize_depth(inv_depth0_shift)
                    inv_depth_vis1_shift = visualization.visualize_depth(inv_depth1_shift)
                    inv_depth_vis0 = visualization.visualize_depth(inv_depth0)
                    inv_depth_vis1 = visualization.visualize_depth(inv_depth1)
                    
                    suffix = "_eps2_{}_gdt_{}_dmm_{}".format(self.opt.eps2, self.opt.g1_diff_threshold, self.opt.depth_mean_min)
                    cv2.imwrite(self.opt.path + 'depth_gt_predicted_png{}/depth_{}_gt_predicted_{}_{}_shift.png'.format(suffix, index1, index1, index2), inv_depth_vis0_shift)
                    cv2.imwrite(self.opt.path + 'depth_gt_predicted_png{}/depth_{}_gt_predicted_{}_{}_shift.png'.format(suffix, index2, index1, index2), inv_depth_vis1_shift)
                    cv2.imwrite(self.opt.path + 'depth_gt_predicted_png{}/depth_{}_gt_predicted_{}_{}.png'.format(suffix, index1, index1, index2), inv_depth_vis0)
                    cv2.imwrite(self.opt.path + 'depth_gt_predicted_png{}/depth_{}_gt_predicted_{}_{}.png'.format(suffix, index2, index1, index2), inv_depth_vis1)
                    cv2.imwrite(self.opt.path + 'depth_gt_predicted_png{}/depth_{}_gt_predicted_{}_{}_masked.png'.format(suffix, index1, index1, index2), inv_depth_vis0*mask0)
                    cv2.imwrite(self.opt.path + 'depth_gt_predicted_png{}/depth_{}_gt_predicted_{}_{}_masked.png'.format(suffix, index2, index1, index2), inv_depth_vis1*mask1)
                        
                    cv2.imwrite(self.opt.path + 'mask_rectified{}/mask_{}_{}.png'.format(suffix, str(index1).zfill(6), str(index2).zfill(6)), mask0 * 255)
                    cv2.imwrite(self.opt.path + 'mask_rectified{}/mask_{}_{}.png'.format(suffix, str(index2).zfill(6), str(index1).zfill(6)), mask1 * 255)
                    if mask0.sum() > 0 and mask1.sum() > 0:
                        if np.isinf(G10_np.max()) or np.isnan(G10_np.max()) or np.isinf(G11_np.max()) or np.isnan(G11_np.max()):
                            pdb.set_trace()                      
                        save_raw_float32_image(self.opt.path + 'depth_gt_predicted{}/depth_{}_gt_predicted_{}_{}.raw'.format(suffix, str(index1).zfill(6), str(index1).zfill(6), str(index2).zfill(6)), G10_np[None, ...])
                        save_raw_float32_image(self.opt.path + 'depth_gt_predicted{}/depth_{}_gt_predicted_{}_{}.raw'.format(suffix, str(index2).zfill(6), str(index1).zfill(6), str(index2).zfill(6)), G11_np[None, ...])
        else:
            if self.opt.gt_prediction_prepro:
                gt_predicted_pair = (gt_predictions for gt_predictions in geom_meta["gt_predictions"])
                masks_prepro_pair = (masks for masks in geom_meta["masks_prepro"])
                depth_projection_median_confidence_map_pair = (masks for masks in geom_meta["depth_projection_median_confidence_map"])
                for (
                    points_cam_ref,
                    tgt_points_cam_tgt,
                    pixels_ref,
                    flows_ref,
                    masks_ref,
                    gt_predictions_ref,
                    masks_prepro_ref,
                    depth_projection_median_confidence_map_ref,
                    intrinsics_ref,
                    intrinsics_tgt,
                    extrinsics_ref,
                    extrinsics_tgt,
                ) in zip(
                    points_cam_pair,
                    points_cam_pair[inv_idxs],
                    pixels_pair,
                    flows_pair,
                    masks_pair,
                    gt_predicted_pair,
                    masks_prepro_pair,
                    depth_projection_median_confidence_map_pair,
                    intrinsics_pair,
                    intrinsics_pair[inv_idxs],
                    extrinsics_pair,
                    extrinsics_pair[inv_idxs],
                ):
                    # change to camera space for target_camera
                    # points_cam_tgt = reproject_points(points_cam_ref, extrinsics_ref, extrinsics_tgt)
                    points_cam_tgt, pos_cam_ref_tgt = reproject_points2(points_cam_ref, extrinsics_ref, extrinsics_tgt)
                    matched_pixels_tgt = pixels_ref + flows_ref # flow-warped
                    pixels_tgt = project(points_cam_tgt, intrinsics_tgt) # depth-warped
                    
                    if self.opt.lambda_reprojection > 0:
                        reproj_dist = torch.norm(pixels_tgt - matched_pixels_tgt,
                            dim=1, keepdim=True)
                        reproj_losses.append(
                            weighted_mean_loss(self.dist(reproj_dist), masks_ref)
                        )

                    if self.opt.lambda_view_baseline > 0:
                        # disparity consistency
                        f = torch.mean(focal_length(intrinsics_ref))
                        # warp points in target image grid target camera coordinates to
                        # reference image grid
                        warped_tgt_points_cam_tgt = sample(
                            tgt_points_cam_tgt, matched_pixels_tgt
                        )

                        disp_diff = 1.0 / points_cam_tgt[:, -1:, ...] \
                            - 1.0 / warped_tgt_points_cam_tgt[:, -1:, ...]
                        
                        disp_losses.append(
                            f * weighted_mean_loss(self.dist(disp_diff), masks_ref)
                        )

                        if self.opt.straight_line_method:
                            points_world_ref, cam_world_ref = to_worldspace(points_cam_ref, extrinsics_ref)
                            points_world_warped_tgt, cam_world_tgt = to_worldspace(warped_tgt_points_cam_tgt, extrinsics_tgt)

                            G2 = gt_predictions_ref
                            if 0: # distance
                                D2 = torch.norm(points_world_warped_tgt - cam_world_tgt, dim=1, keepdim=True)    
                            else: # real depth
                                D2 = torch.abs(warped_tgt_points_cam_tgt[:, 2:, ...])    
                            ref_tgt_diff = torch.norm(points_world_ref - points_world_warped_tgt, dim=1, keepdim=True)

                            if self.opt.loss_case == 231:
                                # median gt predictions
                                if not self.opt.ablation_median_didabled:
                                    if self.opt.gt_prediction_prepro:
                                        if self.opt.confidence_enabled:
                                            depth_losses_g2d2 = self.opt.pred_ref_gt_diff_weight * weighted_mean_loss(self.dist(torch.log10(self.opt.alpha + self.opt.gamma * G2) - torch.log10(self.opt.alpha + self.opt.gamma * D2)), (masks_prepro_ref).float() * depth_projection_median_confidence_map_ref)
                                        else:
                                            depth_losses_g2d2 = self.opt.pred_ref_gt_diff_weight * weighted_mean_loss(self.dist(torch.log10(self.opt.alpha + self.opt.gamma * G2) - torch.log10(self.opt.alpha + self.opt.gamma * D2)), (masks_prepro_ref).float())
                                    else:
                                            depth_losses_g2d2 = self.opt.pred_ref_gt_diff_weight * weighted_mean_loss(self.dist(torch.log10(self.opt.alpha + self.opt.gamma * G2) - torch.log10(self.opt.alpha + self.opt.gamma * D2)), (masks_ref).float())
                                else: # just work for batch size as 1
                                    B, candidates_num, H, W = masks_prepro_ref.shape
                                    cnt = 0
                                    for k in range(candidates_num):
                                        depth_losses_g2d2_t = self.opt.pred_ref_gt_diff_weight * weighted_mean_loss(self.dist(torch.log10(self.opt.alpha + self.opt.gamma * G2[:, :, k, :, :]) - torch.log10(self.opt.alpha + self.opt.gamma * D2)), (masks_prepro_ref[:, k:k+1, :, :]).float())
                                        if cnt == 0:
                                            depth_losses_g2d2 = depth_losses_g2d2_t
                                            cnt += 1
                                        else:
                                            depth_losses_g2d2 += depth_losses_g2d2_t
                                    depth_losses_g2d2 = depth_losses_g2d2 / candidates_num

                                depth_losses_d1d2 = self.opt.ref_tgt_depth_diff_weight * weighted_mean_loss(self.dist(ref_tgt_diff), (masks_ref).float())
                                
                                depth_losses.append(depth_losses_g2d2 + depth_losses_d1d2)
                            else:
                                pass
            else: # 
                gt_predicted_pair = (gt_predictions for gt_predictions in geom_meta["gt_predictions"])
                for (
                    points_cam_ref,
                    tgt_points_cam_tgt,
                    pixels_ref,
                    flows_ref,
                    masks_ref,
                    gt_predictions_ref,
                    intrinsics_ref,
                    intrinsics_tgt,
                    extrinsics_ref,
                    extrinsics_tgt,
                ) in zip(
                    points_cam_pair,
                    points_cam_pair[inv_idxs],
                    pixels_pair,
                    flows_pair,
                    masks_pair,
                    gt_predicted_pair,
                    intrinsics_pair,
                    intrinsics_pair[inv_idxs],
                    extrinsics_pair,
                    extrinsics_pair[inv_idxs],
                ):
                    # change to camera space for target_camera
                    # points_cam_tgt = reproject_points(points_cam_ref, extrinsics_ref, extrinsics_tgt)
                    points_cam_tgt, pos_cam_ref_tgt = reproject_points2(points_cam_ref, extrinsics_ref, extrinsics_tgt)
                    matched_pixels_tgt = pixels_ref + flows_ref # flow-warped
                    pixels_tgt = project(points_cam_tgt, intrinsics_tgt) # depth-warped
                    
                    if self.opt.lambda_reprojection > 0:
                        reproj_dist = torch.norm(pixels_tgt - matched_pixels_tgt,
                            dim=1, keepdim=True)
                        reproj_losses.append(
                            weighted_mean_loss(self.dist(reproj_dist), masks_ref)
                        )

                    if self.opt.lambda_view_baseline > 0:
                        # disparity consistency
                        f = torch.mean(focal_length(intrinsics_ref))
                        # warp points in target image grid target camera coordinates to
                        # reference image grid
                        warped_tgt_points_cam_tgt = sample(
                            tgt_points_cam_tgt, matched_pixels_tgt
                        )

                        disp_diff = 1.0 / points_cam_tgt[:, -1:, ...] \
                            - 1.0 / warped_tgt_points_cam_tgt[:, -1:, ...]
                        
                        disp_losses.append(
                            f * weighted_mean_loss(self.dist(disp_diff), masks_ref)
                        )

                        if self.opt.straight_line_method:
                            points_world_ref, cam_world_ref = to_worldspace(points_cam_ref, extrinsics_ref)
                            points_world_warped_tgt, cam_world_tgt = to_worldspace(warped_tgt_points_cam_tgt, extrinsics_tgt)

                            G2 = gt_predictions_ref
                            D2 = torch.norm(points_world_warped_tgt - cam_world_tgt, dim=1, keepdim=True)                        
                            ref_tgt_diff = torch.norm(points_world_ref - points_world_warped_tgt, dim=1, keepdim=True)

                            if self.opt.loss_case == 231:
                                depth_losses_g2d2 = self.opt.pred_ref_gt_diff_weight * weighted_mean_loss(self.dist(torch.log10(self.opt.alpha + self.opt.gamma * G2) - torch.log10(self.opt.alpha + self.opt.gamma * D2)), (masks_ref.unsqueeze(1)).float())
                                depth_losses_d1d2 = self.opt.ref_tgt_depth_diff_weight * weighted_mean_loss(self.dist(ref_tgt_diff), (masks_ref).float())
                                
                                if torch.isnan(D2.max()) or torch.isnan(G2.max()) or torch.isinf(D2.max()) or torch.isinf(G2.max()):
                                    pdb.set_trace()
                                depth_losses.append(depth_losses_g2d2 + depth_losses_d1d2)
                            else:
                                pass

        B = points_cam_pair[0].shape[0]
        dtype = points_cam_pair[0].dtype

        if self.opt.straight_line_method:
            if self.opt.test_mode:
                # G1 is not reliable, discard this pair of frames by setting depth_losses as 0
                depth_losses_post = []
                for loss_i in depth_losses:
                    depth_losses_post.append(loss_i * 0)
                depth_losses = depth_losses_post

        reproj_loss = (
            self.opt.lambda_reprojection
            * torch.mean(torch.stack(reproj_losses, dim=-1), dim=-1)
            if len(reproj_losses) > 0
            else torch.zeros(B, dtype=dtype, device=_device)
        )
        disp_loss = (
            self.opt.lambda_view_baseline
            * torch.mean(torch.stack(disp_losses, dim=-1), dim=-1)
            if (len(disp_losses) > 0)
            else torch.zeros(B, dtype=dtype, device=_device)
        )
        if self.opt.straight_line_method:
            depth_loss = 1.0 * (
                torch.mean(torch.stack(depth_losses, dim=-1), dim=-1)
                if len(depth_losses) > 0
                else torch.zeros(B, dtype=dtype, device=_device)
            )
            if self.opt.flow_finetune:
                if not self.opt.flow_finetune_epi:
                    g1_g2_loss = 1.0 * (
                        torch.mean(torch.stack(g1_g2_losses, dim=-1), dim=-1)
                        if len(g1_g2_losses) > 0
                        else torch.zeros(B, dtype=dtype, device=_device)
                    )
                    flow_loss = g1_g2_loss
                else:
                    epipolar_loss = 1.0 * (
                        torch.mean(torch.stack(epipolar_losses, dim=-1), dim=-1)
                        if len(epipolar_losses) > 0
                        else torch.zeros(B, dtype=dtype, device=_device)
                    )
                    flow_loss = epipolar_loss

                    
        if self.opt.straight_line_method:
            if self.opt.flow_finetune:
                if self.opt.is_flow_finetune:
                    depth_loss = depth_loss.detach()
                else:
                    flow_loss = flow_loss.detach()
                
                if self.opt.g1_g2_diff_weight != 0 or self.opt.flow_finetune_epi:
                    batch_losses = {"depth": depth_loss, "flow": flow_loss}
                else:
                    batch_losses = {"depth": depth_loss}
                
                return torch.mean(depth_loss + flow_loss), batch_losses
            else:
                batch_losses = {"reprojection": reproj_loss, "disparity": disp_loss, "depth": depth_loss}
                return torch.mean(depth_loss), batch_losses
        else:
            batch_losses = {"reprojection": reproj_loss, "disparity": disp_loss}
            return torch.mean(self.opt.reproj_weight * reproj_loss + self.opt.disp_weight * disp_loss), batch_losses

    def __call__(
        self,
        depths,
        metadata,
    ):
        """Compute total loss.

        The network predicts a set of depths results. The number of samples, N, is
        not the batch_size, but computed based on the loss.
        For instance, geometry_consistency_loss requires pairs as samples, then
            N = 2 .
        If with more losses, say triplet one from temporal_consistency_loss. Then
            N = 2 + 3.

        Args:
            depths (B, N, H, W):   predicted_depths
            metadata: dictionary of related metadata to compute the loss. Here assumes
                metadata include data as below. But each loss assumes more.
                {
                    'extrinsics': torch.tensor (B, N, 3, 4), # extrinsics of each frame.
                                    Each (3, 4) = [R, t]
                    'intrinsics': torch.tensor (B, N, 4),
                                  # (fx, fy, cx, cy) for each frame in pixels
                }

        Returns:
            loss: python scalar. And set self.total_loss
        """

        def squeeze(x):
            return x.reshape((-1,) + x.shape[2:])

        def unsqueeze(x, N):
            return x.reshape((-1, N) + x.shape[1:])

        depths = depths.unsqueeze(-3)
        intrinsics = metadata["intrinsics"]
        B, N, C, H, W = depths.shape
        pixels = pixel_grid(B * N, (H, W))

        points_cam = pixels_to_points(squeeze(intrinsics), squeeze(depths), pixels)
        pixels = unsqueeze(pixels, N)
        points_cam = unsqueeze(points_cam, N)

        return self.geometry_consistency_loss(points_cam, metadata, pixels)
