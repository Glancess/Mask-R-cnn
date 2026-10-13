import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import maskrcnn_loss


class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, result, targets):
        # result：模型训练时返回的字典。
        cls_logits = result["cls_logits"]  # [R, C]
        bbox_deltas = result["bbox_deltas"]  # [R, C*4]

        # 每张图片的标签、回归目标拼在一起。
        labels = torch.cat(result["labels"], dim=0)  # [R]
        regression_targets = torch.cat(result["regression_targets"], dim=0)  # [R, 4]

        # 1. 分类：前景和背景都参与。
        loss_cls = F.cross_entropy(cls_logits, labels)

        # 2. 回归：先把每个类别的四个偏移量分开。
        num_rois = labels.shape[0]
        bbox_deltas = bbox_deltas.reshape(num_rois, -1, 4)
        # [R, C, 4]

        # 找到前景 RoI 的下标，以及它们的真实类别。
        fg_idx = torch.where(labels > 0)[0]
        fg_labels = labels[fg_idx]

        # 每个前景 RoI，只取真实类别对应的四个预测偏移量。
        pred_boxes = bbox_deltas[fg_idx, fg_labels]  # [F, 4]
        target_boxes = regression_targets[fg_idx]  # [F, 4]

        loss_bbox = (
            F.smooth_l1_loss(
                pred_boxes,
                target_boxes,
                beta=1 / 9,
                reduction="sum",
            )
            / num_rois
        )

        """
              "mask_logits": mask_logits,
                        "mask_proposals": mask_proposals,
                        "mask_matched_idxs": mask_matched_idxs,
                        "mask_labels": mask_labels,
        """
        # targets：每张图片一个字典。
        gt_masks = [target["masks"] for target in targets]
        gt_labels = [target["labels"] for target in targets]

        loss_mask = maskrcnn_loss(
            result["mask_logits"],
            result[
                "mask_proposals"
            ],  # proposal 用来告诉函数：应该从整张 GT mask 的哪个区域裁剪出监督目标。
            gt_masks,
            gt_labels,
            result["mask_matched_idxs"],
        )
        # 返回检测分类、框回归、Mask 三项 loss。
        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_mask": loss_mask,
        }
