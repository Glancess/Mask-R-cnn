import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


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

        # 3. Mask：只用前景，且忽略 VOC 中没有可靠标注的像素。
        loss_mask = self.compute_mask_loss(result, targets)
        # 返回检测分类、框回归、Mask 三项 loss。
        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_mask": loss_mask,
        }

    def compute_mask_loss(self, result, targets):
        """
        输入 result：模型返回的预测、前景 proposal 和匹配 GT 下标。
        输入 targets：每张图片的 GT masks、labels、mask_valid。
        返回：一个可反向传播的标量 Tensor。
        mask_valid 是像素是否有效，不是物体类别，也不是前景框标签。
        """
        mask_logits = result["mask_logits"]  # [F,C,28,28]，未经 sigmoid。
        output_size = mask_logits.shape[-1]
        all_mask_targets = []
        all_valid_targets = []
        all_mask_labels = []

        for i in range(len(targets)):
            proposals = result["mask_proposals"][i]  # [Fi,4]，这张图的前景框。
            matched_idxs = result["mask_matched_idxs"][i]  # [Fi]，GT 实例下标。
            if len(proposals) == 0:
                continue

            # 每个 GT 实例的整图 mask 当作一张单通道图：[Mi,1,H,W]。
            gt_masks = targets[i]["masks"].unsqueeze(1).to(proposals)
            # RoIAlign 每行：[从第几个 GT mask 取区域, x1,y1,x2,y2]。
            mask_rois = torch.cat([matched_idxs[:, None].to(proposals), proposals], dim=1)
            mask_targets = roi_align(
                gt_masks, mask_rois, (output_size, output_size),
                spatial_scale=1.0, sampling_ratio=-1, aligned=False,
            )[:, 0]  # [Fi,28,28]；目标与预测覆盖同一个 proposal 区域。

            # 每张原图只有一张有效像素图：[H,W]，VOC 的 255 位置为 False。
            if "mask_valid" in targets[i]:
                valid_map = targets[i]["mask_valid"]
            else:
                # PennFudan 没有忽略区域，全部像素有效，兼容原数据接口。
                valid_map = torch.ones_like(targets[i]["masks"][0], dtype=torch.bool)
            valid_map = valid_map[None, None].to(proposals)  # [1,1,H,W]。
            # 这里只能从第0张图取区域，不能拿 GT 实例下标来索引有效像素图。
            valid_rois = torch.cat([proposals.new_zeros((len(proposals), 1)), proposals], dim=1)
            valid_targets = roi_align(
                valid_map, valid_rois, (output_size, output_size),
                spatial_scale=1.0, sampling_ratio=-1, aligned=False,
            )[:, 0]
            # 插值可能得到小数。保守地只保留所有采样都有效的位置，
            # 避免 255 区域被填成0后，在边界混入监督目标。
            valid_targets = valid_targets >= 1.0 - 1e-6

            all_mask_targets.append(mask_targets)
            all_valid_targets.append(valid_targets)
            all_mask_labels.append(targets[i]["labels"][matched_idxs])

        # 整个 batch 没有前景，也返回连接计算图的零，而不是 Python 数字0。
        if len(all_mask_targets) == 0:
            return mask_logits.sum() * 0

        mask_targets = torch.cat(all_mask_targets, dim=0)  # [F,28,28]。
        valid_targets = torch.cat(all_valid_targets, dim=0)  # bool [F,28,28]。
        mask_labels = torch.cat(all_mask_labels, dim=0)  # [F]，真实物体类别。
        if not valid_targets.any():
            return mask_logits.sum() * 0

        # 每个前景框只取真实类别对应的预测通道。
        indices = torch.arange(len(mask_labels), device=mask_labels.device)
        pred_masks = mask_logits[indices, mask_labels]  # [F,28,28]。
        pixel_loss = F.binary_cross_entropy_with_logits(
            pred_masks, mask_targets, reduction="none"
        )  # 不立即平均，先得到每个像素的 loss。
        return pixel_loss[valid_targets].mean()  # 只对有效像素平均。
