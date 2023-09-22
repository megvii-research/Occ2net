import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

# from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine, PE3
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
from .backbone.global_feature import build_shufflenetv2_backbone as build_global
from .utils.gemo import ReProj

class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.pos_encoding = PositionEncodingSine(
            config["coarse"]["d_model"], temp_bug_fix=config["coarse"]["temp_bug_fix"]
        )
        self.loftr_coarse = LocalFeatureTransformer(config["coarse"])
        self.coarse_matching = CoarseMatching(config["match_coarse"])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()
        self.rotate_feature = config["rotate_feature"]
        self.d3_pe = PE3()
        self.global_net = build_global(config["global_step"], "./shufflenetv2.pth")
        self.occ = nn.Linear(16, 2)

    def forward(self, data):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 0. Global Feature CNN
        # 1. Local Feature CNN
        # faster & better BN convergence
        feat_c0, feat_f0, voxel0, vf0 = self.global_net(data["image0"])
        feat_c1, feat_f1, voxel1, vf1 = self.global_net(data["image1"])
        # here we got an estimate depth, used to instead the best gt occpancy model
        feat_g0, feat_g1 = ReProj(data["depth0"], data["depth1"], data["K0"], data["K1"], data["T_0to1"], data["T_1to0"])
        occ_f0 = voxel0[:,None] * vf0[:,:,None] # N C16 D64 H W
        occ_f1 = voxel1[:,None] * vf1[:,:,None] # N C D H W
        occ_f0 = self.d3_pe(occ_f0)
        occ_f1 = self.d3_pe(occ_f1)
        N, C, D, H, W = occ_f0.shape
        occ_f0 = self.occ(occ_f0.reshape(N, C, D*H*W).permute(0,2,1)).reshape(N, D, H, W, 2)
        occ_f1 = self.occ(occ_f1.reshape(N, C, D*H*W).permute(0,2,1)).reshape(N, D, H, W, 2)
        occ_f0 = torch.softmax(occ_f0, -1)[..., 0]
        occ_f1 = torch.softmax(occ_f1, -1)[..., 0]
        ## loss calculate this ,without scale
        data["occ_f0"] = occ_f0.clone()
        data["occ_f1"] = occ_f1.clone()
        data["occ_g0"] = feat_g0.clone()
        data["occ_g1"] = feat_g1.clone()
        feat_f0 = torch.cat([feat_f0, occ_f0], 1)
        feat_f1 = torch.cat([feat_f1, occ_f1], 1)

        data["image0"] = (
            data["image0"][:, 0] * 0.299
            + data["image0"][:, 1] * 0.587
            + data["image0"][:, 2] * 0.114
        )[:, None]
        data["image1"] = (
            data["image1"][:, 0] * 0.299
            + data["image1"][:, 1] * 0.587
            + data["image1"][:, 2] * 0.114
        )[:, None]
        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )
        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_f0.shape[2:],
                "hw1_f": feat_f1.shape[2:],
            }
        )

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]

        feat_c0 = self.pos_encoding(feat_c0)
        feat_c1 = self.pos_encoding(feat_c1)

        B, C, H, W = feat_c0.shape
        device = feat_c0.device

        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"].flatten(-2), data["mask1"].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        feat_c0 = rearrange(feat_c0, "b (h w) c -> b c h w", h=H, w=W)
        feat_c1 = rearrange(feat_c1, "b (h w) c -> b c h w", h=H, w=W)

        if self.rotate_feature:
            feat_c0_rotate = feat_c0.clone() / 5
            feat_c1_rotate = feat_c1.clone() / 5
            for i in range(4):
                theta = torch.tensor(torch.pi / 6 + i * torch.pi / 2)
                translation = torch.tensor(
                    [
                        [1, 0, 2 / W * torch.cos(theta)],
                        [0, 1, 2 / H * torch.sin(theta)],
                    ],
                    device=device,
                ).repeat(B, 1, 1)
                grid = F.affine_grid(translation, feat_c0.size(), align_corners=False)
                feat_c0_rotate += F.grid_sample(feat_c0, grid, align_corners=False) / 5
            for i in range(4):
                theta = torch.tensor(torch.pi / 6 + i * torch.pi / 2)
                translation = torch.tensor(
                    [
                        [1, 0, 2 / W * torch.cos(theta)],
                        [0, 1, 2 / H * torch.sin(theta)],
                    ],
                    device=device,
                ).repeat(B, 1, 1)
                grid = F.affine_grid(translation, feat_c1.size(), align_corners=False)
                feat_c1_rotate += F.grid_sample(feat_c1, grid, align_corners=False) / 5

        # 3. match coarse-level
        feat_c0 = rearrange(feat_c0, "b c h w -> b (h w) c")
        feat_c1 = rearrange(feat_c1, "b c h w -> b (h w) c")
        if self.rotate_feature:
            feat_c1_rotate = rearrange(feat_c1_rotate, "b c h w -> b (h w) c")
            feat_c0_rotate = rearrange(feat_c0_rotate, "b c h w -> b (h w) c")
            self.coarse_matching(
                feat_c0,
                feat_c1,
                data,
                mask_c0=mask_c0,
                mask_c1=mask_c1,
                feat_c0_rotate=feat_c0_rotate,
                feat_c1_rotate=feat_c1_rotate,
            )
        else:
            self.coarse_matching(
                feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1
            )

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(
            feat_f0, feat_f1, feat_c0, feat_c1, data
        )
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(
                feat_f0_unfold, feat_f1_unfold
            )

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
