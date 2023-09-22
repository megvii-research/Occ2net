import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import pickle
import glob
import cv2
import tqdm
import sys

sys.path.append("..")
from ..utils.position_encoding import PositionEncodingSine


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


class ConvBlock(nn.Module):
    def __init__(
        self,
        ic,
        oc,
        stride=1,
        padding=1,
        kernel_size=3,
        use_bn=True,
        use_relu=True,
        use_bias=False,
        out_scale=None,
    ):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=ic,
                out_channels=oc,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=use_bias,
            )
        )
        if use_bn:
            self.conv.add_module("BN", nn.BatchNorm2d(oc))
        if use_relu:
            self.conv.add_module("ReLU", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup // 2

        branch_main = [
            ConvBlock(
                ic=inp,
                oc=mid_channels,
                kernel_size=ksize,
                stride=stride,
                padding=1,
                use_bias=False,
            ),
            ConvBlock(
                ic=mid_channels,
                oc=outputs,
                kernel_size=ksize,
                stride=1,
                padding=1,
                use_bias=False,
            ),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                ConvBlock(
                    ic=inp,
                    oc=outputs,
                    kernel_size=ksize,
                    stride=stride,
                    padding=1,
                    use_bias=False,
                ),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            shuffle_conv_a = [
                ConvBlock(
                    ic=int(inp * 2),
                    oc=inp,
                    kernel_size=ksize,
                    stride=1,
                    padding=1,
                    use_bias=False,
                )
            ]
            shuffle_conv_b = [
                ConvBlock(
                    ic=int(inp * 2),
                    oc=outputs,
                    kernel_size=ksize,
                    padding=1,
                    stride=1,
                    use_bias=False,
                )
            ]
            self.shuffle_conv_a = nn.Sequential(*shuffle_conv_a)
            self.shuffle_conv_b = nn.Sequential(*shuffle_conv_b)
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj = self.shuffle_conv_a(old_x)
            x = self.shuffle_conv_b(old_x)
            return torch.cat([x_proj, self.branch_main(x)], 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat([self.branch_proj(x_proj), self.branch_main(x)], 1)


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        input_size=224,
        n_class=1000,
        model_version="v7",
        out_features=None,
    ):
        super(ShuffleNetV2, self).__init__()
        print("model version is ", model_version)

        self.model_version = model_version
        if model_version == "v11":
            self.stage_repeats = [1, 1, 4, 8, 4]
            self.stage_out_channels = [-1, 64, 128, 256, 256, 256, 512]
            self.stage_ends_idx = [2 - 1, 6 - 1, 14 - 1, 18 - 1]
        else:
            raise NotImplementedError

        self._out_feature_strides = {
            "res1": 2,
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
            "res6": 64,
        }
        self._out_feature_channels = {}
        for i, c in enumerate(self.stage_out_channels[1:-1], 1):
            self._out_feature_channels["res" + str(i)] = c
        self._out_features = out_features
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = ConvBlock(
            ic=3,
            oc=input_channel,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=False,
        )
        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(
                        ShuffleV2Block(
                            input_channel,
                            output_channel,
                            mid_channels=output_channel,
                            ksize=3,
                            stride=2,
                        )
                    )
                else:
                    self.features.append(
                        ShuffleV2Block(
                            input_channel // 2,
                            output_channel,
                            mid_channels=output_channel,
                            ksize=3,
                            stride=1,
                        )
                    )

                input_channel = output_channel

        self.features0 = nn.Sequential(*self.features[:1])
        self.features1 = nn.Sequential(*self.features[1:2])
        self.features2 = nn.Sequential(*self.features[2:6])
        self.pos_encoding = PositionEncodingSine()
        # self._initialize_weights()
        self.layer0_outconv = conv1x1(128, 256)
        self.layer0_outconv2 = conv1x1(64, 256)
        self.layer0_outconv3 = conv1x1(128, 256)
        self.layer2_outconv = conv1x1(256, 256)
        self.layer2_outconv3 = conv1x1(256, 64)
        self.layer3_voxel = conv1x1(256, 64)
        self.layer3_feature = conv1x1(256, 16)

    def forward(self, x):
        x = self.first_conv(x)  # 1/2
        x0 = self.features0(x)
        x0 = self.pos_encoding(x0)  # 1/4
        x1 = self.features1(x0)  # 1/8
        x2 = self.features2(x1)  # 1/16

        x2_out = F.interpolate(
            x2, scale_factor=2.0, mode="bilinear", align_corners=True
        )  # 1/8
        x0_out = F.interpolate(
            x0, scale_factor=0.5, mode="bilinear", align_corners=True
        )  # 1/8
        x0_out = self.layer0_outconv(x0_out)
        out_coarse = self.layer2_outconv(x0_out + x1 + x2_out)

        x_c = self.layer0_outconv2(x)  # 1/2
        x0_c = F.interpolate(x0, scale_factor=2, mode="bilinear", align_corners=True)
        x0_c = self.layer0_outconv3(x0_c)
        x1_c = F.interpolate(x1, scale_factor=4, mode="bilinear", align_corners=True)
        out_fine = self.layer2_outconv3(x_c + x1_c + x0_c)

        out_voxel = self.layer3_voxel(x_c + x1_c + x0_c)
        out_vf = self.layer3_feature(x_c + x1_c + x0_c)
        return out_coarse, out_fine, out_voxel, out_vf


def build_shufflenetv2_backbone(out_features, model_path):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    out_features = out_features
    pretrain_model = model_path
    model_version = "v11"

    model = ShuffleNetV2(model_version=model_version, out_features=out_features)

    print("-------------------------------------------------------------")
    print("load weights from {}".format(pretrain_model))
    with open(model_path, "rb") as model_file:
        checkpoint = pickle.load(model_file)

    state_dict = checkpoint["model"]
    new_state_dict = dict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    return model
