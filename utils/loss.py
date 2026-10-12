import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


def box_loss(class_logits, box_regression, labels_list, bbox_targets_list):
    """
    输入 class_logits: float32[R,C]；box_regression: float32[R,C*4]。
    labels_list: list[int64[Ri]]；bbox_targets_list: list[float32[Ri,4]]。
    返回 dict：loss_classifier / loss_box_reg，都是可反传的标量 Tensor。
    C 包含背景；R = sum(Ri)。
    """
    labels = torch.cat(labels_list, dim=0)
    bbox_targets = torch.cat(bbox_targets_list, dim=0)

    if len(labels) == 0:
        return {
            "loss_classifier": class_logits.sum() * 0,
            "loss_box_reg": box_regression.sum() * 0,
        }

    # 1. 前景、背景都参与分类；CrossEntropyLoss 内部处理 softmax。
    cls_loss = F.cross_entropy(class_logits, labels)

    # 2. 只回归正样本，而且只选这个正样本真实类别的4个偏移量。
    # [R,C*4] -> [R,C,4]；这里包含背景通道，所以用 labels，不要 labels-1！
    box_regression = box_regression.reshape(len(labels), -1, 4)
    positive = torch.where(labels > 0)[0]
    bbox_loss = F.smooth_l1_loss(
        box_regression[positive, labels[positive]],
        bbox_targets[positive],
        beta=1 / 9,
        reduction="sum",
    ) / len(labels)

    return {"loss_classifier": cls_loss, "loss_box_reg": bbox_loss}


def mask_loss(mask_logits, positive_proposals, positive_labels, positive_matched_idxs, targets):
    """
    输入 mask_logits: float32[P,C,28,28]，只对正样本预测的原始 logits。
    其余输入是长度 B 的 list：正样本框[Pi,4]、类别[Pi]、GT实例下标[Pi]、target。
    返回 loss_mask: 可反传的标量 Tensor。
    GT 整图 mask -> 按正 proposal 裁剪 -> 28x28；再与真实类别通道比较。
    """
    mask_targets = []

    for boxes, matched_idxs, target in zip(positive_proposals, positive_matched_idxs, targets):
        if len(boxes) == 0:
            continue

        # 1. 把每个 GT 实例暂时当作 roi_align 输入里的一个 batch 元素。
        # [M,H,W] -> [M,1,H,W]，不是把不同实例合成一张语义分割图。
        gt_masks = target["masks"].to(dtype=boxes.dtype)[:, None]

        # 2. [Pi,5] 的第一列是本图 GT 实例编号；后4列是采样 proposal 的 xyxy。
        # 例如 [2,x1,y1,x2,y2] 表示从 gt_masks[2] 裁剪，不是第2张图片。
        rois = torch.cat([matched_idxs[:, None].to(boxes.dtype), boxes], dim=1)
        cropped_masks = roi_align(
            gt_masks, rois, output_size=mask_logits.shape[-2:],
            spatial_scale=1.0, sampling_ratio=2,
        )[:, 0]
        mask_targets.append(cropped_masks)

    # 无前景时仍返回连在计算图上的0，而不是新建一个不能反传的常量。
    if len(mask_targets) == 0:
        return mask_logits.sum() * 0

    mask_targets = torch.cat(mask_targets, dim=0)  # [P,28,28]，范围[0,1]
    labels = torch.cat(positive_labels, dim=0)     # [P]
    indices = torch.arange(len(labels), device=labels.device)

    # 3. 每个正 RoI 只监督其真实类别的 mask 通道；背景不算 mask loss。
    # BCEWithLogits 内部做 sigmoid，不能先 sigmoid 再传给这个函数。
    return F.binary_cross_entropy_with_logits(mask_logits[indices, labels], mask_targets)
