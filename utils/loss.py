import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from torchvision.models.detection.roi_heads import maskrcnn_loss


class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, result, targets):
        # result：模型训练时返回的字典。

        """
        return {
                    "firstLoss": first_result["firstLoss"],
                    "cls_logits": cls_logits,
                    "bbox_deltas": bbox_deltas,
                    "labels": labels,
                    "regression_targets": regression_targets,
                    "mask_logits": mask_logits,
                    "mask_proposals": mask_proposals,
                    "mask_matched_idxs": mask_matched_idxs,
                    "mask_labels": mask_labels,
                }
        """
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

        # 3. Mask：只用前景，暂时不处理 VOC 的255忽略区域。
        loss_mask = self.compute_mask_loss(result, targets)
        # 返回检测分类、框回归、Mask 三项 loss。
        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_mask": loss_mask,
        }

    """
     return {
                    "firstLoss": first_result["firstLoss"],
                    "cls_logits": cls_logits,
                    "bbox_deltas": bbox_deltas,
                    "labels": labels,
                    "regression_targets": regression_targets,
                    "mask_logits": mask_logits,
                    "mask_proposals": mask_proposals,
                    "mask_matched_idxs": mask_matched_idxs,
                    "mask_labels": mask_labels,
                }
    """

    def compute_mask_loss(self, result, targets):
        gt_masks = [target["masks"] for target in targets]
        gt_labels = [target["labels"] for target in targets]
        loss_mask = maskrcnn_loss(
            result["mask_logits"],
            result["mask_proposals"],
            gt_masks,
            gt_labels,
            result["mask_matched_idxs"],
        )
        return loss_mask


if __name__ == "__main__":
    loss_fn = MaskLoss()

    # 1 个 positive RoI，21 个类别
    mask_logits = torch.randn(1, 21, 28, 28, requires_grad=True)

    # 当前 positive proposal
    proposals = torch.tensor([[16.0, 16.0, 48.0, 48.0]], dtype=torch.float32)

    # 一张 GT instance mask
    gt_mask = torch.zeros((1, 64, 64), dtype=torch.uint8)
    gt_mask[0, 20:45, 20:45] = 1

    # 当前 proposal 匹配 GT instance 0
    matched_idxs = torch.tensor([0], dtype=torch.int64)

    # 当前 GT 是 person = 15
    gt_labels = torch.tensor([15], dtype=torch.int64)

    # 全图有效区域
    valid = torch.ones((64, 64), dtype=torch.bool)

    # 故意制造一块 VOC 255 区域
    valid[30:40, 30:40] = False

    result = {
        "mask_logits": mask_logits,
        # 注意这里都是“按图片组织”的 list
        "mask_proposals": [proposals],
        "mask_matched_idxs": [matched_idxs],
    }

    targets = [
        {
            "masks": gt_mask,
            "labels": gt_labels,
            "mask_valid": valid,
        }
    ]

    loss_mask = loss_fn.compute_mask_loss(result, targets)

    print("loss_mask:", loss_mask)

    loss_mask.backward()

    print("backward ok")
