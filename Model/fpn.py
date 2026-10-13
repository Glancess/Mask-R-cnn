from .backbone import BackBone

from torchvision.ops import FeaturePyramidNetwork
import torch.nn.functional as F
import torch
import torch.nn as nn


class GetPx(nn.Module):
    def __init__(self):
        super().__init__()
        self.fpn = FeaturePyramidNetwork(
            [256, 512, 1024, 2048],
            256,
        )

    def forward(self, C):
        x = self.fpn(C)
        p5 = F.max_pool2d(x["c4"], kernel_size=1, stride=2)
        x["c5"] = p5
        return x


if __name__ == "__main__":
    image = torch.randn(3, 3, 256, 256)
    model = BackBone()
    fpn = GetPx()
    result = fpn(model(image))

    for name, feature in result.items():
        print(name, feature.shape)
