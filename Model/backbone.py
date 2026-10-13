import torch
import torch.nn as nn
from torchvision.models import resnet50


class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        ren = resnet50()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            ),
        )
        self.c2 = ren.layer1
        self.c3 = ren.layer2
        self.c4 = ren.layer3
        self.c5 = ren.layer4

    def forward(self, x):
        c = self.stem(x)
        c1 = self.c2(c)
        c2 = self.c3(c1)
        c3 = self.c4(c2)
        c4 = self.c5(c3)
        return {
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "c4": c4,
        }


if __name__ == "__main__":

    x = torch.randn(3, 3, 200, 200)
    print(x)
    model = BackBone()
    a = model(x)
    print(type(a))
    for name, feat in a.items():
        print(name, feat.shape)
