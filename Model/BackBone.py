import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Backbone(nn.Module):
    def __init__(self, weights=ResNet50_Weights.DEFAULT):
        """weights: 预训练权重；传 None 可以做不下载权重的接口检查。"""
        super().__init__()

        resnet = resnet50(weights=weights)

        # stem
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # C2 ~ C5
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        """
        输入:
            x: float32 Tensor[B,3,H,W]，已做 ImageNet normalize 的 RGB 图片。
        返回:
            OrderedDict[str, Tensor]，不是一个单独的 Tensor。
            c2: [B, 256,H/4, W/4 ]
            c3: [B, 512,H/8, W/8 ]
            c4: [B,1024,H/16,W/16]
            c5: [B,2048,H/32,W/32]
            上面按 H/W 是32的倍数写；其他尺寸以实际输出为准。
        """

        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # bottom-up
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return OrderedDict(
            {
                "c2": c2,
                "c3": c3,
                "c4": c4,
                "c5": c5,
            }
        )


"""
我们得到resnet50大额 C2-C5，采取一个字典的存放
"""
