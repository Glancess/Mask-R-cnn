from typing import Any

from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
import torch
import torch.nn as nn


class getFirstproposal(nn.Module):
    def __init__(self):
        super().__init__()
        self.rpnhead = RPNHead(256, 3)

        self.anchorgenerate = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        self.firstproposal = RegionProposalNetwork(
            self.anchorgenerate,
            self.rpnhead,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7,
        )

    def forward(self, Imagelist, P, targets=None, Train=True):
        """
                targets = [
            {
                "boxes": torch.tensor(
                    [
                        [20, 30, 100, 150],  # 第一张图的第一个 GT
                        [120, 40, 200, 180], # 第一张图的第二个 GT
                    ],
                    dtype=torch.float32,
                )
            },
            {
                "boxes": torch.tensor(
                    [[40, 50, 160, 220]],
                    dtype=torch.float32,
                )
            },
            {
                "boxes": torch.empty((0, 4), dtype=torch.float32)
            },
        ]"""

        # List[Tensor])是boxes的
        """
        like list[
        image1:[[],[],[]......]
        ]
        image2:[[],[]......]
        """
        # torchvision RPN 根据自己的 training 开关决定是否计算 loss。
        # 验证需要 loss，但不能把整个模型切回 train（会更新 BN）。
        old_training = self.firstproposal.training
        self.firstproposal.training = Train
        try:
            boxes, losses = self.firstproposal(Imagelist, P, targets)
        finally:
            self.firstproposal.training = old_training
        return {"firstLoss": losses, "firstproposal": boxes}
