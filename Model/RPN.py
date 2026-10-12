import torch
import torch.nn as nn

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import (
    RPNHead,
    RegionProposalNetwork,
)
from torchvision.models.detection.image_list import ImageList


class RPNModule(nn.Module):
    def __init__(self):
        super().__init__()

        # --------------------------------------------------
        # 1. Anchor Generator
        #
        # P2 -> 32
        # P3 -> 64
        # P4 -> 128
        # P5 -> 256
        # P6 -> 512
        #
        # 每个位置 3 个 ratio，因此每个位置 3 个 anchor
        # --------------------------------------------------
        self.anchor_generator = AnchorGenerator(
            sizes=(
                (32,),
                (64,),
                (128,),
                (256,),
                (512,),
            ),
            aspect_ratios=(
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
                (0.5, 1.0, 2.0),
            ),
        )

        # --------------------------------------------------
        # 2. RPN Head
        #
        # 每个 FPN level:
        #
        # [B,256,H,W]
        #       ↓
        #     3x3 conv
        #       ↓
        # ┌───────────────┬────────────────┐
        # objectness      bbox regression
        # [B,3,H,W]       [B,12,H,W]
        #
        # 3 = 3 anchors
        # 12 = 3 anchors × 4 bbox deltas
        # 这里每个 anchor 只有1个物体 logit，内部用 sigmoid/BCE。
        # 与之前 Faster 项目的2个分类 logit + CE 不同，两种都可用于二分类。
        # --------------------------------------------------
        self.rpn_head = RPNHead(
            in_channels=256,
            num_anchors=3,
        )

        # --------------------------------------------------
        # 3. 完整 RPN
        # --------------------------------------------------
        self.rpn = RegionProposalNetwork(
            anchor_generator=self.anchor_generator,
            head=self.rpn_head,
            # anchor 与 GT 的 IoU 匹配
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            # 训练 RPN loss 时，每张图采样 256 个 anchor
            batch_size_per_image=256,
            # 正样本最多占 50%
            positive_fraction=0.5,
            # NMS 之前保留的高分 proposal
            pre_nms_top_n={
                "training": 2000,
                "testing": 1000,
            },
            # NMS 之后最终保留的 proposal
            post_nms_top_n={
                "training": 2000,
                "testing": 1000,
            },
            nms_thresh=0.7,
        )

    def forward(
        self,
        images,
        pyramid,
        targets=None,
        image_sizes=None,
        compute_loss=None,
    ):
        """
        images:
            float32 Tensor [B, 3, H_pad, W_pad]，补零后的图片。

        pyramid:
            {
                "p2": Tensor[B,256,H2,W2],
                "p3": Tensor[B,256,H3,W3],
                "p4": Tensor[B,256,H4,W4],
                "p5": Tensor[B,256,H5,W5],
                "p6": Tensor[B,256,H6,W6],
            }

        targets:
            类型: list[dict[str, Tensor]]，长度B；推理时是None。
            训练时:
            [
                {
                    "boxes": float32 Tensor[N1, 4]，xyxy,
                    ...
                },
                {
                    "boxes": Tensor[N2, 4],
                    ...
                },
            ]

            推理时:
            None

        image_sizes:
            list[tuple[int,int]]，每张图片真实的 (H, W)，不能填统一补零尺寸。

            如果所有图片大小一样，可以不传，
            自动从 images.shape 获取。

        compute_loss:
            bool或None。None跟随self.training；True即使验证也返回RPN loss。
            仅切换内部RPN的行为，不会把外面的ResNet切回训练模式。

        return:
            tuple(proposals, losses)
            proposals:
                list[Tensor]

                proposals[i]:
                    第 i 张图片的 proposals
                    shape = [Ni, 4]

            losses:
                dict[str, Tensor]，每项loss是标量Tensor，可以参与backward。
                训练:
                {
                    "loss_objectness": ...,
                    "loss_rpn_box_reg": ...
                }

                推理:
                {}
        """

        B, _, H, W = images.shape

        if compute_loss is None:
            compute_loss = self.training

        if compute_loss and targets is None:
            raise ValueError("RPN训练需要targets；只检查预测时请先调用rpn.eval()。")

        # 如果 batch 中图片全部一样大，可以直接推断
        if image_sizes is None:
            image_sizes = [(H, W)] * B

        # torchvision RPN 需要 ImageList
        image_list = ImageList(images, image_sizes)

        # 注意：
        # pyramid 直接传 dict 即可
        # RegionProposalNetwork 内部会自己转换：
        #
        # features = list(features.values())
        #
        # 然后再传给 RPNHead
        # torchvision 用 training 标志决定是否计算 RPN loss，以及候选框上限。
        # 验证 loss 时用训练版候选框配置，但不更新参数；调用结束恢复原标志。
        # RPN head 没有 BN/Dropout，不需要把整个模型切回 train()。
        previous_training = self.rpn.training
        self.rpn.training = compute_loss
        try:
            proposals, losses = self.rpn(image_list, pyramid, targets)
        finally:
            self.rpn.training = previous_training

        return proposals, losses  # 已经得到初步修过的框


if __name__ == "__main__":
    # 在项目目录运行：python -m Model.RPN
    from .BackboneWithFPN import BackboneWithFPN

    backbone = BackboneWithFPN(weights=None)
    rpn = RPNModule()

    backbone.eval()
    rpn.eval()

    x = torch.randn(3, 3, 128, 128)

    with torch.no_grad():

        # Image -> ResNet50 -> FPN
        pyramid = backbone(x)

        for name, feat in pyramid.items():
            print(name, feat.shape)

        # FPN -> RPN -> proposals
        proposals, losses = rpn(
            images=x,
            pyramid=pyramid,
        )
    print(type(proposals))
    print(len(proposals))
    print(proposals[0].shape)

    print("\nproposal数量：")

    for i, boxes in enumerate(proposals):
        print(f"image {i}:", boxes.shape)

    print("losses:", losses)

"""
anchors
↓
和 target["boxes"] 算 IoU
↓
给 anchor 打标签
positive / negative / ignore
↓
按 batch_size_per_image 和 positive_fraction 采样
↓
计算：
1. objectness loss
2. bbox regression loss
↓
同时另一条路径生成 proposals
"""
