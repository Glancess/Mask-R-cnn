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
                 target = {
                            "boxes": boxes,  # [N, 4]
                            "labels": labels,  # [N]
                            "masks": masks,  # [N, H, W]
                        }
                
                        return image, target
    """

    def compute_mask_loss(self, result, targets):
        # 取出整个batch的Mask预测，还没有经过sigmoid。
        mask_logits = result["mask_logits"]  # float Tensor，[F,C,S,S]

        # 取预测mask的宽度，当前为28。
        mask_size = mask_logits.shape[-1]  # Python int，S=28

        # 三个列表分别保存：监督mask、有效位置、真实类别。
        all_mask_targets = []  # list，每个元素是[Fi,S,S]
        all_valid_targets = []  # list，每个元素是bool [Fi,S,S]
        all_mask_labels = []  # list，每个元素是int64 [Fi]

        # 遍历batch里的每张图片，i表示当前图片的编号。
        for i in range(len(targets)):

            # 取当前图片的所有GT实例mask，每个实例是一张整图二值图。
            gt_masks_i = targets[i]["masks"]  # uint8 Tensor，[Mi,H,W]

            # 取当前图片的所有GT实例类别，与上面的实例顺序一致。
            gt_labels_i = targets[i]["labels"]  # int64 Tensor，[Mi]

            # 取当前图片抽样后的前景框，坐标格式为xyxy。
            proposals = result["mask_proposals"][i]  # float Tensor，[Fi,4]

            # 取每个前景框匹配的GT实例编号。
            matched_idxs = result["mask_matched_idxs"][i]  # int64 Tensor，[Fi]

            # 1. 生成GT mask的监督目标。

            # 添加单通道维度，再转成浮点数，供RoIAlign采样。
            gt_masks_i = gt_masks_i.unsqueeze(1).float()  # [Mi,1,H,W]

            # 把每个框匹配的GT编号变成一列，并转为浮点数。
            gt_indices = matched_idxs[:, None].float()  # [Fi,1]

            # 每行：[GT实例编号,x1,y1,x2,y2]。
            # 第一列指定从哪一个GT mask裁剪，后四列指定裁剪区域。
            rois = torch.cat([gt_indices, proposals], dim=1)  # [Fi,5]

            # 按每个前景框，从对应GT mask提取固定大小的监督目标。
            # 坐标和整图mask使用相同尺度，所以spatial_scale=1.0。
            mask_target_i = roi_align(
                gt_masks_i,  # 输入：[Mi,1,H,W]
                rois,  # 框：[Fi,5]
                output_size=(mask_size, mask_size),
                spatial_scale=1.0,
            )  # 输出：float Tensor，[Fi,1,S,S]

            # 取出唯一的通道，保留每个前景框的二维目标。
            mask_target_i = mask_target_i[:, 0]  # [Fi,S,S]

            # 保存当前图片的监督目标，列表中的顺序就是图片顺序。
            all_mask_targets.append(mask_target_i)

            # 2. 生成与上面监督目标对应的有效区域。

            # 取当前图片的有效区域图：True有效，False表示需要忽略。
            valid_map = targets[i]["mask_valid"].float()  # [H,W]，转成0/1浮点数

            # 当前只有一张有效区域图、一个通道。
            valid_map = valid_map.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

            # 复制同一批裁剪框，保持后四列的坐标完全相同。
            valid_rois = rois.clone()  # [Fi,5]

            # 有效区域图只有一张，所以所有框都从第0张图裁剪。
            valid_rois[:, 0] = 0  # shape仍为[Fi,5]

            # 用相同的区域、输出尺寸和采样设置，裁剪有效区域图。
            valid_target_i = roi_align(
                valid_map,  # 输入：[1,1,H,W]
                valid_rois,  # 框：[Fi,5]
                output_size=(mask_size, mask_size),
                spatial_scale=1.0,
            )  # 输出：float Tensor，[Fi,1,S,S]

            # 取出唯一的通道。
            valid_target_i = valid_target_i[:, 0]  # [Fi,S,S]

            # 插值可能产生小数，只保留接近1、没有明显混入无效区域的位置。
            valid_target_i = valid_target_i >= 1.0 - 1e-6  # bool Tensor，[Fi,S,S]

            # 保存当前图片的有效区域。
            all_valid_targets.append(valid_target_i)

            # 3. 查询每个前景框对应的真实类别。

            # matched_idxs选择GT实例，再从该实例取出类别。
            # 例如GT类别[1,15]、匹配下标[0,0,1]，得到[1,1,15]。
            mask_labels_i = gt_labels_i[matched_idxs]  # int64 Tensor，[Fi]

            # 保存当前图片每个前景框的真实类别。
            all_mask_labels.append(mask_labels_i)

        # 4. 按图片顺序拼接，顺序与模型的Mask预测一致。

        # 拼接整个batch的监督目标。
        mask_target = torch.cat(all_mask_targets, dim=0)  # float Tensor，[F,S,S]

        # 拼接整个batch的有效区域。
        valid_target = torch.cat(all_valid_targets, dim=0)  # bool Tensor，[F,S,S]

        # 拼接整个batch的前景框类别。
        mask_labels = torch.cat(all_mask_labels, dim=0)  # int64 Tensor，[F]

        # 5. 每个前景框，只选择其真实类别对应的预测通道。

        # 生成前景框编号：[0,1,...,F-1]，与预测放在同一设备。
        indices = torch.arange(
            len(mask_labels), device=mask_logits.device
        )  # int64 Tensor，[F]

        # 第r个框取mask_logits[r, mask_labels[r]]。
        # 从每框C个通道中选出一个通道。
        pred_masks = mask_logits[indices, mask_labels]  # float Tensor，[F,S,S]

        # 6. 计算逐像素BCE，再筛选有效位置。

        # 不提前sigmoid，BCEWithLogits内部会完成相应计算。
        # reduction="none"表示暂时不求平均，保留每个位置的loss。
        pixel_loss = F.binary_cross_entropy_with_logits(
            pred_masks, mask_target, reduction="none"
        )  # float Tensor，[F,S,S]

        # 布尔索引只留下有效位置，结果变成一维。
        valid_loss = pixel_loss[valid_target]  # float Tensor，[K]，K是有效位置数

        # 只对有效位置求平均，得到可反向传播的标量。
        loss_mask = valid_loss.mean()  # float Tensor，shape为[]

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
