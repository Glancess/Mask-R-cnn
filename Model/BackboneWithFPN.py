import torch
from .BackBone import ResNet50Backbone
from torchvision.models import ResNet50_Weights
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict
import torch.nn.functional as F


class BackboneWithFPN(nn.Module):
    def __init__(self, weights=ResNet50_Weights.DEFAULT):
        super().__init__()

        self.backbone = ResNet50Backbone(weights=weights)

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048], out_channels=256
        )

    def forward(self, x):
        """
        输入 x: float32 Tensor[B,3,H,W]，已经 normalize 的图片。
        返回 pyramid: OrderedDict[str, Tensor]。
            p2/p3/p4/p5/p6 的通道数都是256，stride分别为4/8/16/32/64。
            例如输入 [B,3,256,256]，p2是[B,256,64,64]，p6是[B,256,4,4]。
        """

        # C2-C5
        features = self.backbone(x)

        # features 是有序字典；FPN 按 c2 -> c5 的顺序读取不同尺度特征。
        # P2-P5
        pyramid = self.fpn(features)

        pyramid = OrderedDict(
            {
                "p2": pyramid["c2"],
                "p3": pyramid["c3"],
                "p4": pyramid["c4"],
                "p5": pyramid["c5"],
            }
        )
        p6 = F.max_pool2d(pyramid["p5"], kernel_size=1, stride=2)
        pyramid["p6"] = p6
        return pyramid


"""
我们根据算子FPN，得到 处理后的C2-C5，同时加入P6. 继续返回了一个字典。
"""

if __name__ == "__main__":
    # 在项目目录运行：python -m Model.BackboneWithFPN
    model = BackboneWithFPN(weights=None).eval()

    x = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        pyramid = model(x)

    for name, feat in pyramid.items():
        print(name, feat.shape)
