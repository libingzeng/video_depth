#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn

import pdb

def sample(data, uv):
    """Sample data (H, W, <C>) by uv (H, W, 2) (in pixels). """
    shape = data.shape
    # data from (H, W, <C>) to (1, C, H, W)
    # data = data.reshape(data.shape[:2] + (-1,))
    # data = torch.tensor(data).permute(2, 0, 1)[None, ...]
    data = data.permute(0, 3, 1, 2)
    # (H, W, 2) -> (1, H, W, 2)
    # uv = torch.tensor(uv)[None, ...]

    N, H, W, _ = shape
    # grid needs to be in [-1, 1] and (B, H, W, 2)
    size = torch.tensor((W, H), dtype=uv.dtype).view(1, 1, 1, -1).cuda()
    grid = (2 * uv / size - 1).to(data.dtype)
    tensor = torch.nn.functional.grid_sample(data, grid, padding_mode="border")
    # from (1, C, H, W) to (H, W, <C>)
    return tensor.permute(0, 2, 3, 1).reshape(shape)


def sse(x, y, dim=-1):
    """Sum of suqare error"""
    d = x - y
    return torch.sum(d * d, dim=dim)


def consistency_mask(im_ref, im_tgt, flow, threshold, diff_func=sse):
    im_ref = torch.tensor(im_ref).cuda()
    im_tgt = torch.tensor(im_tgt).cuda()
    flow = torch.tensor(flow).cuda()
    if len(flow.shape) == 3:
        im_ref = im_ref[None, :]
        im_tgt = im_tgt[None, :]
        flow = flow[None, :]
    
    N, H, W, _ = im_ref.shape
    # im_ref = im_ref.reshape(H, W, -1)
    # im_tgt = im_tgt.reshape(H, W, -1)
    x = torch.linspace(0, W - 1, W).cuda()
    y = torch.linspace(0, H - 1, H).cuda()
    Y, X = torch.meshgrid(y, x)

    idx_x = X + flow[..., 0]
    idx_y = Y + flow[..., 1]

    # first constrain to within the image
    mask = torch.all(
        torch.stack((idx_x >= 0, idx_x <= W - 1, 0 <= idx_y, idx_y <= H - 1), dim=-1),
        dim=-1,
    )

    im_tgt_to_ref = sample(im_tgt, torch.stack((idx_x, idx_y), dim=-1))

    # mask = torch.logical_and(mask, diff_func(im_ref, im_tgt_to_ref) < threshold)
    mask = mask * (diff_func(im_ref, im_tgt_to_ref) < threshold)
    
    return mask


def consistent_flow_masks(flows, colors, flow_thresh, color_thresh):
    # mask from flow consistency
    masks_flow = [
        consistency_mask(flow_ref, -flow_tgt, flow_ref, flow_thresh ** 2)
        for flow_ref, flow_tgt in zip(flows, flows[::-1])
    ]
    # mask from photometric consistency
    C = colors[0].shape[-1]
    masks_photo = [
        consistency_mask(c_ref, c_tgt, flow_ref, C * (color_thresh ** 2))
        for c_ref, c_tgt, flow_ref in zip(colors, colors[::-1], flows)
    ]
    # merge the two
    masks = [(mf * mp)[0].cpu().numpy() for mf, mp in zip(masks_flow, masks_photo)]
    # masks = [(mf * mp) for mf, mp in zip(masks_flow, masks_photo)]
    
    return masks
