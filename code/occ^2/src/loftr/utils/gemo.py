import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def ReProj(depth0, depth1, K0, K1, T01, T10, C=64):
    depth0_f1, depth1_f0 = warp_depth(depth0, depth1, T01, T10, K0, K1)
    depth0_f1 = depth0_f1[:,::2,::2].clone()
    depth1_f0 = depth1_f0[:,::2,::2].clone()
    depth0 = depth0[:,::2,::2].clone()
    depth1 = depth1[:,::2,::2].clone()
    B, H, W = depth0.shape
    max_v = max(depth0.max(), depth1.max())
    min_v = min(depth0.min(), depth1.min())
    depth0_f1 = torch.clamp(depth0_f1, min_v, max_v)
    depth1_f0 = torch.clamp(depth1_f0, min_v, max_v)
    depth0 = ((depth0 - min_v) / (max_v - min_v) * 63).int()
    depth1 = ((depth1 - min_v) / (max_v - min_v) * 63).int()
    mask0 = (depth0_f1 != 0)
    mask1 = (depth1_f0 != 0)
    depth0_f1 = ((depth0_f1 - min_v) / (max_v - min_v) * 63).int()
    depth1_f0 = ((depth1_f0 - min_v) / (max_v - min_v) * 63).int()
    x_position = torch.ones(H, W).cumsum(1).int().unsqueeze(0) - 1 
    y_position = torch.ones(H, W).cumsum(0).int().unsqueeze(0) - 1
    voxel0 = torch.zeros((B, C, H, W)).to(depth0.device)
    voxel1 = torch.zeros((B, C, H, W)).to(depth1.device)
    voxel0[:, depth0_f1[mask0].long(), y_position[mask0].long(), x_position[mask0].long()] = 1
    voxel1[:, depth1_f0[mask1].long(), y_position[mask1].long(), x_position[mask1].long()] = 1
    voxel0[:, depth0.long(), y_position.long(), x_position.long()] = 1
    voxel1[:, depth1.long(), y_position.long(), x_position.long()] = 1
    return voxel0, voxel1

@torch.no_grad()
def warp_depth(depth0, depth1, T_0to1, T_1to0, K0, K1):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).

    Args:
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        warped_depth0_from_depth1[N, H, W]
        warped_depth1_from_depth0[N,H,W]
    """
    B, H, W = depth0.shape
    x_position = (torch.ones(H, W).cumsum(1).int().unsqueeze(0) - 1).to(depth0.device)
    y_position = (torch.ones(H, W).cumsum(0).int().unsqueeze(0) - 1).to(depth0.device)

    # Unproject
    kpts0_h = (
        torch.cat([x_position.reshape(B,-1,1), y_position.reshape(B,-1,1), depth0.reshape(B,-1,1)], dim=-1)
    )# (N, L, 3)
    kpts1_h = (
        torch.cat([x_position.reshape(B,-1,1), y_position.reshape(B,-1,1), depth1.reshape(B,-1,1)], dim=-1)
    )# (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1) # (N, 3, L) 
    kpts1_cam = K1.inverse() @ kpts1_h.transpose(2, 1) # (N, 3, L) 

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :] # depth1_from 0

    w_kpts1_cam = T_1to0[:, :3, :3] @ kpts1_cam + T_1to0[:, :3, [3]]  # (N, 3, L)
    w_kpts1_depth_computed = w_kpts1_cam[:, 2, :]   # depth0 from 1
    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth
    # depth1 from0, xy postions

    w_kpts1_h = (K0 @ w_kpts1_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts1 = w_kpts1_h[:, :, :2] / (
        w_kpts1_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth
    # depth 0 from 1, xy positions

    # 这个范围好像不是01的，
    depth1_f0 = backwarp(w_kpts1_depth_computed.reshape(B, 1, H, W), w_kpts1.reshape(B, H, W, 2).permute(0, 3, 1, 2))
    depth0_f1 = backwarp(w_kpts0_depth_computed.reshape(B, 1, H, W), w_kpts0.reshape(B, H, W, 2).permute(0, 3, 1, 2))
    return depth0_f1[:,0], depth0_f1[:,0]

def backwarp(tenInput, tenFlow):
    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    g = tenFlow.permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
