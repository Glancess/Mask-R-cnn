import torch
from torchvision.models.detection.roi_heads import paste_masks_in_image


def postprocess_boxes(class_logits, box_regression, proposals, image_shapes, roi_sampler):
    """
    输入 logits[R,C]、bbox delta[R,C*4]、每图 proposals[Ni,4]、真实尺寸列表。
    roi_sampler: 你的 RoISampler，用它内部同一个 BoxCoder 解码。
    返回 list[dict]，每张图 boxes[Di,4] / scores[Di] / labels[Di]。
    """
    # 与训练目标使用相同的 10/10/5/5 权重。
    # 黑盒完成：softmax → 解码修框 → 裁到图内 → 去背景 → 分数过滤 → 按类别NMS。
    boxes, scores, labels = roi_sampler.roi_heads.postprocess_detections(
        class_logits, box_regression, proposals, image_shapes,
    )
    results = []
    for boxes_i, scores_i, labels_i in zip(boxes, scores, labels):
        results.append({"boxes": boxes_i, "scores": scores_i, "labels": labels_i})
    return results


def restore_predictions(results, mask_logits, image_shapes, original_shapes):
    """
    results: list[dict]，缩放后的最终检测框；mask_logits: Tensor[D,C,28,28]。
    两个 shapes 参数: list[(H,W)]，缩放后尺寸 / 原图尺寸。
    返回 list[dict]，框恢复原坐标，并增加 masks: float32[Di,1,H,W] 概率图。
    """
    counts = [len(result["boxes"]) for result in results]
    per_image_masks = mask_logits.split(counts, dim=0)
    restored = []
    for result, logits, resized_shape, original_shape in zip(
        results, per_image_masks, image_shapes, original_shapes,
    ):
        h, w = resized_shape
        original_h, original_w = original_shape
        ratio = result["boxes"].new_tensor([original_w / w, original_h / h] * 2)
        boxes = result["boxes"] * ratio

        # 每个检测只取预测类别的 mask 通道，不使用 GT 类别。
        indices = torch.arange(len(logits), device=logits.device)
        masks = logits[indices, result["labels"]].sigmoid()[:, None]  # [Di,1,28,28]

        # 黑盒：局部28×28 mask缩放到对应框，再贴回原图；空检测也支持。
        masks = paste_masks_in_image(masks, boxes, original_shape)
        restored.append({
            "boxes": boxes, "scores": result["scores"],
            "labels": result["labels"], "masks": masks,
        })
    return restored
