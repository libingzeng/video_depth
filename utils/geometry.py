#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn.functional as F
from .torch_helpers import _device
from typing import List

import pdb

def pixel_grid(batch_size, shape):
    """Returns pixel grid of size (batch_size, 2, H, W).
    pixel positions (x, y) are in range [0, W-1] x [0, H-1]
    top left is (0, 0).
    """
    H, W = shape
    x = torch.linspace(0, W - 1, W, device=_device)
    y = torch.linspace(0, H - 1, H, device=_device)
    Y, X = torch.meshgrid(y, x)
    pixels = torch.stack((X, Y), dim=0)[None, ...]
    return pixels.expand(batch_size, -1, -1, -1)


def principal_point(intrinsics, shape):
    """
    Args:
        intrinsics: (fx, fy, cx, cy)
        shape: (H, W)
    """
    return intrinsics[:, 2:]
    # # center version
    # H, W = shape
    # return torch.tensor(((W - 1) / 2.0, (H - 1) / 2.0), device=_device)


def focal_length(intrinsics):
    return intrinsics[:, :2]


def pixels_to_rays(pixels, intrinsics):
    """Convert pixels to rays in camera space using intrinsics.

    Args:
        pixels (B, 2, H, W)
        intrinsics (B, 4): (fx, fy, cx, cy)

    Returns:
        rays: (B, 3, H, W), where z component is -1, i.e., rays[:, -1] = -1

    """
    # Assume principal point is ((W-1)/2, (H-1)/2).
    B, _, H, W = pixels.shape
    cs = principal_point(intrinsics, (H, W))
    # Convert to [-(W-1)/2, (W-1)/2] x [-(H-1)/2, (H-1)/2)] and bottom left is (0, 0)
    uvs = pixels - cs.view(-1, 2, 1, 1)
    uvs[:, 1] = -uvs[:, 1]  # flip v

    # compute rays (u/fx, v/fy, -1)
    fxys = focal_length(intrinsics).view(-1, 2, 1, 1)
    rays = torch.cat(
        (uvs / fxys, -torch.ones((B, 1, H, W), dtype=uvs.dtype, device=_device)), dim=1
    )
    return rays


def pixels_to_homogeneous_coord(pixels, intrinsics):
    """Convert pixels to homogeneous coordinates in camera space using intrinsics.

    Args:
        pixels (B, 2, H, W)
        intrinsics (B, 4): (fx, fy, cx, cy)

    Returns:
        coord: (B, 3, H, W), where z component is 1

    """
    # Assume principal point is ((W-1)/2, (H-1)/2).
    B, _, H, W = pixels.shape
    cs = principal_point(intrinsics, (H, W))
    # Convert to [-(W-1)/2, (W-1)/2] x [-(H-1)/2, (H-1)/2)] and bottom left is (0, 0)
    uvs = pixels - cs.view(-1, 2, 1, 1)
    uvs[:, 1] = -uvs[:, 1]  # flip v

    # compute rays (u/fx, v/fy, 1)
    fxys = focal_length(intrinsics).view(-1, 2, 1, 1)
    coord = torch.cat(
        (uvs / fxys, torch.ones((B, 1, H, W), dtype=uvs.dtype, device=_device)), dim=1
    )
    # pdb.set_trace()
    return coord


def project(points, intrinsics):
    """Project points in camera space to pixel coordinates based on intrinsics.
    Args:
        points (B, 3, H, W)
        intrinsics (B, 4): (fx, fy, cx, cy)

    Returns:
        pixels (B, 2, H, W)
    """
    rays = points / -points[:, -1:]
    # rays in pixel unit
    fxys = focal_length(intrinsics)
    uvs = rays[:, :2] * fxys.view(-1, 2, 1, 1)

    B, _, H, W = uvs.shape
    cs = principal_point(intrinsics, (H, W))
    # to pixels: (i, j) = (u, -v) + (cx, cy)
    uvs[:, 1] = -uvs[:, 1]  # flip v
    pixels = uvs + cs.view(-1, 2, 1, 1)
    return pixels


def pixels_to_points(intrinsics, depths, pixels):
    """Convert pixels to 3D points in camera space. (Camera facing -z direction)

    Args:
        intrinsics:
        depths (B, 1, H, W)
        pixels (B, 2, H, W)

    Returns:
        points (B, 3, H, W)

    """
    rays = pixels_to_rays(pixels, intrinsics)
    points = rays * depths
    return points


def to_worldspace(points_cam, extrinsics):
    """Reproject points in camera coordinate to world space

    Args:
        points_cam (B, 3, H, W): points in camera coordinate.
        extrinsics (B, 3, 4): [R, t] of camera.

    Returns:
        points_world (B, 3, H, W): points in world space.
        camera_pos   (B, 3, H, W): camera position in world space, which is tiled in the same shape of points_world.

    """
    B, p_dim, H, W = points_cam.shape
    assert p_dim == 3, "dimension of point {} != 3".format(p_dim)

    # t + R * p where t of (B, 3, 1), R of (B, 3, 3) and p of (B, 3, H*W)
    R = extrinsics[..., :p_dim]
    t = extrinsics[..., -1:]
    points_world = torch.baddbmm(t, R, points_cam.view(B, p_dim, -1))
    camera_pos = t.repeat(1, 1, points_world.shape[2])

    return points_world.view(B, p_dim, H, W), camera_pos.view(B, p_dim, H, W)


def to_camera(points_world, extrinsics):
    """map points in world space to camera space

    Args:
        points_world (B, 3, H, W): points in world space.
        extrinsics (B, 3, 4): [R, t] of camera.

    Returns:
        points_camera (B, 3, H, W): points in camera space.

    """
    B, p_dim, H, W = points_world.shape
    assert p_dim == 3, "dimension of point {} != 3".format(p_dim)
    
    # map to camera:
    # R'^T * (p - t') where t' of (B, 3, 1), R' of (B, 3, 3) and p of (B, 3, H*W)
    R_tgt = extrinsics[..., :p_dim]
    t_tgt = extrinsics[..., -1:]
    points_cam = torch.bmm(R_tgt.transpose(1, 2), points_world.view(B, p_dim, -1) - t_tgt)

    return points_cam.view(B, p_dim, H, W)


def ends_common_perpendicular(cam_ref_tgt, points_ref_tgt, cam_tgt, points_tgt, image_mask, eps=1e-6, eps2=1e-6, max_depth=1e3, cam_dist=0.0, parallel_lines_out=False):
    """ending points of common perpendicular of corresponding rays of reference pixel and target pixel

    Args:
        cam_ref_tgt (B, 3, H, W): reference camera position.
        points_ref_tgt (B, 3, H, W): rays of reference pixels.
        cam_tgt (B, 3, H, W): target camera position.
        points_tgt (B, 3, H, W): rays of target pixels.

    Returns:
        ending_ref_tgt (B, 3, H, W): ending point of common perpendicular which is on reference ray.
        ending_tgt (B, 3, H, W): ending point of common perpendicular which is on target ray.
    
    """
    B, p_dim, H, W = points_ref_tgt.shape

    vec1 = F.normalize(points_ref_tgt - cam_ref_tgt, dim=1, p=2)
    vec2 = F.normalize(points_tgt - cam_tgt, dim=1, p=2)
    
    a = torch.unsqueeze(torch.sum(vec1 * vec2, dim=1), 1)
    b = torch.unsqueeze(torch.sum(vec1 * vec1, dim=1), 1)
    c = torch.unsqueeze(torch.sum(vec2 * vec2, dim=1), 1)
    d = torch.unsqueeze(torch.sum(vec1 * (cam_tgt - cam_ref_tgt), dim=1), 1)
    e = torch.unsqueeze(torch.sum(vec2 * (cam_tgt - cam_ref_tgt), dim=1), 1)
    # torch.norm(cam_tgt - cam_ref_tgt, dim=1, keepdim=True).max()
    # (t1_clone == MAX_DEPTH).nonzero().shape
    # (torch.norm(cam_tgt - cam_ref_tgt, dim=1, keepdim=True) > 0.01).nonzero().shape

    EPSILON = eps
    EPSILON_T = eps2 # 1e-2
    MAX_DEPTH = max_depth
    denominator1 = (b * c - a * a).clone()
    denominator1[torch.abs(denominator1) <= EPSILON] = EPSILON

    t1 = (d * c - a * e) / (denominator1)
    t2 = (a * t1 - e) / c

    t1_clone = t1.clone()
    t2_clone = t2.clone()
    t1_clone[denominator1 == EPSILON] = MAX_DEPTH# 10# 
    t2_clone[denominator1 == EPSILON] = MAX_DEPTH# 10# 
    # t1_clone[denominator1 == EPSILON] = t1_clone[denominator1 > EPSILON].max() 
    # t2_clone[denominator1 == EPSILON] = t2_clone[denominator1 > EPSILON].max() 

    t1_v = (t1_clone * vec1).clone()
    t2_v = (t2_clone * vec2).clone()
    ending_ref_tgt = cam_ref_tgt + t1_v
    ending_tgt = cam_tgt + t2_v

    mask_depth = torch.ones(B, 1, H, W).cuda()
    if parallel_lines_out:
        mask_depth[(t1_clone <= EPSILON_T) + (t2_clone <= EPSILON_T) + (denominator1 == EPSILON)] = 0.
    else:
        mask_depth[(t1_clone <= EPSILON_T) + (t2_clone <= EPSILON_T)] = 0.

    cam_diff = torch.norm(cam_tgt - cam_ref_tgt, dim=1, keepdim=True)
    mask_parallel = (cam_diff > cam_dist)

    # pdb.set_trace()

    # if (torch.any(t1_clone * image_mask == MAX_DEPTH) or torch.any(t1_clone * image_mask < 0)):
    # # if (torch.any(t1_clone * image_mask == MAX_DEPTH) or torch.any(t1_clone * image_mask < 0)) and ~torch.any(mask_parallel == 0):
    # # if ((t1_clone * image_mask == MAX_DEPTH).sum() > 100 or (t1_clone * image_mask < 0).sum() > 100) and ~torch.any(mask_parallel == 0):
    #     pdb.set_trace()
    #     import numpy as np
    #     np.savetxt('t1_clone.txt', (t1_clone * image_mask)[0, 0, :, :].cpu().detach().numpy())
    
    # if torch.any(mask_parallel == 0):
    #     pdb.set_trace()
    
    return ending_ref_tgt, ending_tgt, mask_depth * mask_parallel
    # return ending_ref_tgt, ending_tgt, mask_depth

def gt_cvd(cam_ref_tgt, points_ref_tgt, cam_tgt, points_tgt, image_mask, eps=1e-6, eps2=1e-6, depth_mean_min=0.9, max_depth=1e3, cam_dist=0.0, parallel_lines_out=False):
    """ending points of common perpendicular of corresponding rays of reference pixel and target pixel

    Args:
        cam_ref_tgt (B, 3, H, W): reference camera position.
        points_ref_tgt (B, 3, H, W): rays of reference pixels.
        cam_tgt (B, 3, H, W): target camera position.
        points_tgt (B, 3, H, W): rays of target pixels.

    Returns:
        ending_ref_tgt (B, 3, H, W): ending point of common perpendicular which is on reference ray.
        ending_tgt (B, 3, H, W): ending point of common perpendicular which is on target ray.
    
    """
    B, p_dim, H, W = points_ref_tgt.shape

    vec1 = F.normalize(points_ref_tgt - cam_ref_tgt, dim=1, p=2)
    vec2 = F.normalize(points_tgt - cam_tgt, dim=1, p=2)

    vec4 = cam_tgt - cam_ref_tgt
    a = torch.unsqueeze(torch.sum(vec4 * vec1, dim=1), 1)
    b = torch.unsqueeze(torch.sum(vec2 * vec1, dim=1), 1)
    c = torch.unsqueeze(torch.sum(vec4 * vec4, dim=1), 1)
    d = torch.unsqueeze(torch.sum(vec2 * vec2, dim=1), 1)
    e = torch.unsqueeze(torch.sum(vec4 * vec2, dim=1), 1)

    A = b * b * e - a * b * d
    B = b * b * c - a * a * d
    C = a * b * c - a * a * e
    delta = torch.sqrt(B * B - 4 * A * C)
    delta_clone = delta.clone()
    delta_clone[torch.isnan(delta)] = 0

    A[abs(A) < eps] = eps
    t1 = (- B + delta_clone) / (2 * A) # minimum
    t2 = (- B - delta_clone) / (2 * A) # maximum
    
    t1_v = (t1 * vec1).clone()
    t2_v = (t2 * vec2).clone()
    ending_ref_tgt = cam_ref_tgt + t1_v
    ending_tgt = cam_tgt + t2_v

    BN, p_dim, H, W = points_ref_tgt.shape
    mask_depth = torch.ones(BN, 1, H, W).cuda()
    # mask_depth[t2 <= eps] = 0.
    mask_depth[t2 <= eps2] = 0.
    mask_depth[torch.isnan(delta)] = 0.

    cam_diff = torch.norm(cam_tgt - cam_ref_tgt, dim=1, keepdim=True)
    mask_parallel = (cam_diff > cam_dist)

    if t2[(t2>eps2) * (image_mask.float() > 0)].mean() < depth_mean_min:
        mask_depth = torch.zeros(BN, 1, H, W).cuda()
 
    return ending_ref_tgt, ending_tgt, mask_depth * mask_parallel


def gt_spatial_loss_method(cam_ref_tgt, points_ref_tgt, points_image_plane_ref, plane_center_ref, epipole_ref, cam_tgt, points_tgt, image_mask, eps=1e-6, eps2=1e-6, depth_mean_min=0.9, max_depth=1e3, cam_dist=0.0, parallel_lines_out=False):
    """
    Args:
        cam_ref_tgt (B, 3, H, W): reference camera position in world space.
        points_ref_tgt (B, 3, H, W): rays of reference pixels in world space.
        points_image_plane_ref (B, 3, H, W): image plane points of reference camera in world space.
        plane_center_ref (B, 3, H, W): image plane center point of reference camera in world space.
        epipole_ref (B, 3, H, W): epipole on image plane of reference camera in world space.
        cam_tgt (B, 3, H, W): target camera position in world space.
        points_tgt (B, 3, H, W): rays of target pixels in world space.

    Returns:
        ending_ref_tgt (B, 3, H, W): ending point of common perpendicular which is on reference ray.
        ending_tgt (B, 3, H, W): ending point of common perpendicular which is on target ray.
    
    """

    vec1 = F.normalize(points_image_plane_ref - cam_ref_tgt, dim=1, p=2)
    vec2 = F.normalize(points_tgt - cam_tgt, dim=1, p=2)
    vec3 = F.normalize(plane_center_ref - cam_ref_tgt, dim=1, p=2) # z axis of reference camera
    vec4 = F.normalize(cam_tgt - cam_ref_tgt, dim=1, p=2)
    vec5 = F.normalize(torch.cross(vec2, vec4, dim=1), dim=1, p=2) # normal of plane o1o2q
    vec_line = F.normalize(torch.cross(vec3, vec5, dim=1), dim=1, p=2) # vector of epipolar line on the image plane of reference camera
    # pdb.set_trace()
    vec_pe = points_image_plane_ref - epipole_ref 
    vec_pe_line_len = torch.sum(vec_pe * vec_line, dim=1, keepdim=True)
    point_min_spatial_loss = vec_pe_line_len * vec_line + epipole_ref # points on image plane of reference camera that give minimum spatial loss
    # vec6 = F.normalize(point_min_spatial_loss - cam_ref_tgt, dim=1, p=2)
    ending_ref_tgt, ending_tgt, mask_depth = ends_common_perpendicular(cam_ref_tgt, point_min_spatial_loss, cam_tgt, points_tgt, image_mask, eps, eps2, max_depth, cam_dist, parallel_lines_out)

    return ending_ref_tgt, ending_tgt, mask_depth


def loss_epipolar(img1=None, img2=None, flow12=None, intrinsics1=None, intrinsics2=None, extrinsics1=None, extrinsics2=None):
    """1, find the epiline of some point A of img1 in img2 using intrinsics and extrinsics
       2, find the corresponding point A' of A' in img2 using flow12
       3, determine the distance between A' and the epiline

    Args:
        img1: 
        img2:
        flow12: optical flow from img1 to img2
        intrinsics1: 
        intrinsics2: 
        extrinsics1: 
        extrinsics2: 

    Returns:
        loss: the distance between A' and the epiline

    """
    loss = 0

    return loss


def reproject_points(points_cam_ref, extrinsics_ref, extrinsics_tgt):
    """Reproject points in reference camera coordinate to target camera coordinate

    Args:
        points_cam_ref (B, 3, H, W): points in reference camera coordinate.
        extrinsics_ref (B, 3, 4): [R, t] of reference camera.
        extrinsics_tgt (B, 3, 4): [R, t] of target_camera.

    Returns:
        points_cam_tgt (B, 3, H, W): points in target camera coordinate.

    """
    B, p_dim, H, W = points_cam_ref.shape
    assert p_dim == 3, "dimension of point {} != 3".format(p_dim)

    # t + R * p where t of (B, 3, 1), R of (B, 3, 3) and p of (B, 3, H*W)
    R_ref = extrinsics_ref[..., :p_dim]
    t_ref = extrinsics_ref[..., -1:]
    points_world = torch.baddbmm(t_ref, R_ref, points_cam_ref.view(B, p_dim, -1))

    # Reproject to target:
    # R'^T * (p - t') where t' of (B, 3, 1), R' of (B, 3, 3) and p of (B, 3, H*W)
    R_tgt = extrinsics_tgt[..., :p_dim]
    t_tgt = extrinsics_tgt[..., -1:]
    points_cam_tgt = torch.bmm(R_tgt.transpose(1, 2), points_world - t_tgt)
    return points_cam_tgt.view(B, p_dim, H, W)


def reproject_points2(points_cam_ref, extrinsics_ref, extrinsics_tgt):
    """Reproject points in reference camera coordinate to target camera coordinate

    Args:
        points_cam_ref (B, 3, H, W): points in reference camera coordinate.
        extrinsics_ref (B, 3, 4): [R, t] of reference camera.
        extrinsics_tgt (B, 3, 4): [R, t] of target_camera.

    Returns:
        points_cam_tgt (B, 3, H, W): points in target camera coordinate.
        pos_cam_ref_tgt (B, 3, H, W): position of reference camera in target camera coordinate.

    """
    B, p_dim, H, W = points_cam_ref.shape
    assert p_dim == 3, "dimension of point {} != 3".format(p_dim)

    # t + R * p where t of (B, 3, 1), R of (B, 3, 3) and p of (B, 3, H*W)
    R_ref = extrinsics_ref[..., :p_dim]
    t_ref = extrinsics_ref[..., -1:]
    points_world = torch.baddbmm(t_ref, R_ref, points_cam_ref.view(B, p_dim, -1))

    # Reproject to target:
    # R'^T * (p - t') where t' of (B, 3, 1), R' of (B, 3, 3) and p of (B, 3, H*W)
    R_tgt = extrinsics_tgt[..., :p_dim]
    t_tgt = extrinsics_tgt[..., -1:]
    points_cam_tgt = torch.bmm(R_tgt.transpose(1, 2), points_world - t_tgt)
    pos_cam_ref_tgt = torch.bmm(R_tgt.transpose(1, 2), t_ref - t_tgt).repeat(1, 1, points_cam_tgt.shape[2])
    
    return points_cam_tgt.view(B, p_dim, H, W), pos_cam_ref_tgt.view(B, p_dim, H, W)


def depth_to_points(depths, intrinsics):
    """
    Args:
        depths: (B, 1, H, W)
        intrinsics: (B, num_params)
    """
    B, _, H, W = depths.shape
    pixels = pixel_grid(B, (H, W))
    points_cam = pixels_to_points(intrinsics, depths, pixels)
    return points_cam


def calibrate_scale(extrinsics, intrinsics, depths):
    """Given depths, compute the global scale to adjust the extrinsics.
    Given a pair of depths, intrinsics, extrinsics, unproject the depth maps,
    rotate these points based on camera rotation and compute the center for each one.
    The distance between these centers should be of the same scale as the translation
    between the cameras. Therefore, let mu1, mu2 and t1, t2 be the two scene centers
    and the two camera projection centers. Then
        -scale * (t1 - t2) = mu1 - mu2.
    Therefore,
        scale = -dt.dot(dmu) / dt.dot(dt), where dt = t1 - t2, dmu = mu1 - mu2.

    Args:
        intrinsics (2, num_params)
        extrinsics (2, 3, 4): each one is [R, t]
        depths (2, 1, H, W)
    """
    assert (
        extrinsics.shape[0] == intrinsics.shape[0]
        and intrinsics.shape[0] == depths.shape[0]
    )
    points_cam = depth_to_points(depths, intrinsics)
    B, p_dim, H, W = points_cam.shape
    Rs = extrinsics[..., :p_dim]
    ts = extrinsics[..., p_dim]
    points_rot = torch.bmm(Rs, points_cam.view(B, p_dim, -1))
    mus = torch.mean(points_rot, axis=-1)
    # TODO(xuanluo): generalize this to more framse B>2 via variances of the points.
    assert B == 2
    dmu = mus[0] - mus[1]
    dt = ts[0] - ts[1]
    t_scale = -dt.dot(dmu) / dt.dot(dt)
    return t_scale


def warping_field(extrinsics, intrinsics, depths, tgt_ids: List[int]):
    """ Generate the warping field to warp the other frame the current frame.
    Args:
        intrinsics (N, num_params)
        extrinsics (N, 3, 4): each one is [R, t]
        depths (N, 1, H, W)
        tgt_ids (N, 1): warp frame tgt_ids[i] to i

    Returns:
        uvs (N, 2, H, W): sampling the other frame tgt_ids[i] with uvs[i] produces
            the current frame i.
    """
    assert (
        extrinsics.shape[0] == intrinsics.shape[0]
        and intrinsics.shape[0] == depths.shape[0]
    )

    points_cam = depth_to_points(depths, intrinsics)
    extrinsics_tgt = extrinsics[tgt_ids]
    points_tgt_cam = reproject_points(points_cam, extrinsics, extrinsics_tgt)
    uv_tgt = project(points_tgt_cam, intrinsics[tgt_ids])
    return uv_tgt


def sample(data, uv):
    """Sample data (B, C, H, W) by uv (B, 2, H, W) (in pixels). """
    H, W = data.shape[2:]
    # grid needs to be in [-1, 1] and (B, H, W, 2)
    # NOTE: divide by (W-1, H-1) instead of (W, H) because uv is in [-1,1]x[-1,1]
    size = torch.tensor((W - 1, H - 1), dtype=uv.dtype).view(1, -1, 1, 1).to(_device)
    grid = (2 * uv / size - 1).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(data, grid, padding_mode="border")


def warp_image(images, depths, extrinsics, intrinsics, tgt_ids: List[int]):
    """ Warp target images to the reference image based on depths and camera params
    Warp images[tgt_ids[i]] to images[i].

    Args:
        images (N, C, H, W)
        depths (N, 1, H, W)
        extrinsics (N, 3, 4)
        intrinsics (N, 4)
        tgt_ids (N, 1)

    Returns:
        images_warped
    """
    uv_tgt = warping_field(extrinsics, intrinsics, depths, tgt_ids)
    images_warped_to_ref = sample(images[tgt_ids], uv_tgt)
    return images_warped_to_ref
