import torch.nn as nn
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class MaskCOCO(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. 先加载整套COCO权重，不是只加载ResNet50。
        # Backbone、FPN、RPN、检测特征层、Mask特征层都保留预训练参数。
        model = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
            trainable_backbone_layers=5,  # 和手写版一样，Backbone卷积都参与训练。
            box_batch_size_per_image=64,  # 保留你每张图最多抽64个RoI的设置。
        )

        # 2. VOC是20类物体+背景，只有类别相关的输出层需要重新学习。
        # 检测分支：输入[R,1024]，输出分类[R,21]和偏移[R,84]。
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)

        # Mask分支：只换最后的1×1卷积，保留COCO学过的反卷积层。
        # 输入[F,256,28,28]，输出[F,21,28,28]。
        in_channels = model.roi_heads.mask_predictor.mask_fcn_logits.in_channels
        model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(in_channels, 21, 1)

        # 3. 取出官方模块，下面仍按你的顺序一步一步调用。
        self.backbone = model.backbone  # 已经包含ResNet50和FPN。
        self.rpn = model.rpn
        self.roi_heads = model.roi_heads

    def forward(self, imageList, targets=None, Train=True):
        """
        输入：imageList包含[B,3,Hpad,Wpad]和每张图填充前的(H,W)。
              targets是list[dict]，含boxes、labels、masks、mask_valid。
        返回：与手写Mask相同的dict，继续使用原来的train/val/loss/test。
        """
        # DataLoader已经做过归一化和补零，不再调用官方model.transform。
        # 不额外缩放图片，框和mask继续使用当前图片的像素坐标。
        # P：dict，P2～P6的key是"0"、"1"、"2"、"3"、"pool"。
        # 每层的Tensor是[B,256,Hlevel,Wlevel]。
        P = self.backbone(imageList.tensors)

        # 4. RPN：得到第一次修框后的proposals和两项RPN loss。
        # val时model.eval()，但Train=True仍需要RPN计算loss。
        # 只临时切换RPN的开关，不把整个模型切回train。
        old_training = self.rpn.training
        self.rpn.training = Train
        try:
            proposals, first_losses = self.rpn(imageList, P, targets)
        finally:
            self.rpn.training = old_training
        # proposals：list[Tensor]，第i张图片是[Ni,4]，xyxy。

        if Train:
            # 5. 给proposal匹配GT、抽样，得到类别和回归目标。
            Secondproposals, matched_idxs, labels, regression_targets = (
                self.roi_heads.select_training_samples(proposals, targets)
            )

            mask_proposals = []
            mask_matched_idxs = []
            mask_labels = []
            for i in range(len(Secondproposals)):
                fg = labels[i] > 0
                mask_proposals.append(Secondproposals[i][fg])
                mask_matched_idxs.append(matched_idxs[i][fg])
                mask_labels.append(labels[i][fg])
        else:
            # 预测不传GT，也不做训练抽样。
            Secondproposals = proposals

        # 6. 检测分支：前景+背景RoI -> 7×7 -> 两层FC -> 分类和偏移。
        branch1features = self.roi_heads.box_roi_pool(
            P, Secondproposals, imageList.image_sizes
        )  # [R,256,7,7]
        x = self.roi_heads.box_head(branch1features)  # [R,1024]
        cls_logits, bbox_deltas = self.roi_heads.box_predictor(x)
        # cls_logits：[R,21]，bbox_deltas：[R,84]。

        if not Train:
            # 第二次修框、分数筛选、按类别NMS，得到最终检测框。
            boxes, scores, pred_labels = self.roi_heads.postprocess_detections(
                cls_logits, bbox_deltas, Secondproposals, imageList.image_sizes
            )
            mask_proposals = boxes

        # 7. Mask分支：训练只用前景框；预测使用最终检测框。
        branch2features = self.roi_heads.mask_roi_pool(
            P, mask_proposals, imageList.image_sizes
        )  # [F,256,14,14]，预测时第一维是检测框总数D。
        x = self.roi_heads.mask_head(branch2features)  # [F,256,14,14]
        mask_logits = self.roi_heads.mask_predictor(x)  # [F,21,28,28]

        if Train:
            # 仍交给你自己的MaskLoss：分类、前景回归、忽略255的masked BCE。
            return {
                "firstLoss": first_losses,
                "cls_logits": cls_logits,
                "bbox_deltas": bbox_deltas,
                "labels": labels,
                "regression_targets": regression_targets,
                "mask_logits": mask_logits,
                "mask_proposals": mask_proposals,
                "mask_matched_idxs": mask_matched_idxs,
                "mask_labels": mask_labels,
            }

        return {
            "boxes": boxes,  # list[Tensor]，每张图[Di,4]。
            "scores": scores,  # list[Tensor]，每张图[Di]。
            "labels": pred_labels,  # list[Tensor]，每张图[Di]。
            "mask_logits": mask_logits,  # [D,21,28,28]，仍未sigmoid。
        }
