import torch.nn as nn
from torchvision.models.detection.roi_heads import RoIHeads


class RoISampler(nn.Module):
    def __init__(self, num_rois=512, score_thresh=0.5,
                 nms_thresh=0.5, detections_per_img=100):
        """
        输入普通 int/float，保留你用 torchvision 封装的匹配、采样黑盒。
        num_rois 是每张图最多采样多少个 RoI，正样本最多占 25%。
        后三个参数只影响最终预测，不影响训练时 GT 的类别标签。
        """
        super().__init__()

        self.roi_heads = RoIHeads(
            box_roi_pool=None,
            box_head=None,
            box_predictor=None,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=num_rois,
            positive_fraction=0.25,
            # None 的默认值就是这四个数。这里写明白，避免预测解码用错。
            # 第二阶段目标是 [10*dx, 10*dy, 5*dw, 5*dh]。
            bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
        )

    def forward(self, proposals, targets):
        """
        proposals: list[float32 Tensor[Ni,4]]，每张图片的 RPN 框，xyxy。
        targets: list[dict]，至少包含 boxes[Ngt,4]、labels[Ngt]。

        返回四个长度 B 的 list，逐图、逐行对应：
            sampled_proposals[i]: float32[Ri,4]，包含补入的 GT 框。
            matched_idxs[i]: int64[Ri]，匹配的本图 GT 实例下标。
            labels[i]: int64[Ri]，0=背景，1=person。
            regression_targets[i]: float32[Ri,4]，带上述权重的 bbox delta。

        背景也有 matched_idx，但 bbox / mask loss 不使用背景的 GT。
        黑盒内部已经完成 IoU 匹配、补入 GT、前景/背景采样及框编码。
        """

        (
            sampled_proposals,
            matched_idxs,
            labels,
            regression_targets,
        ) = self.roi_heads.select_training_samples(
            proposals,
            targets,
        )

        return (
            sampled_proposals,
            matched_idxs,
            labels,
            regression_targets,
        )
