import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights

from .BackboneWithFPN import BackboneWithFPN
from .RPN import RPNModule
from .RoIHeads import RoIHeads
from utils.RoISampler import RoISampler
from utils.loss import box_loss, mask_loss
from utils.postprocess import postprocess_boxes, restore_predictions
from utils.transform import prepare_images


class MaskRCNN(nn.Module):
    def __init__(self, num_classes=2, weights=ResNet50_Weights.DEFAULT,
                 min_size=400, max_size=640, num_rois=512,
                 score_thresh=0.5, nms_thresh=0.5, detections_per_img=100):
        """
        输入：类别数包含背景；weights=None不下载预训练权重；其余为普通int/float。
        保留你写的BackboneWithFPN、RPNModule、RoIHeads，这里只负责连接它们。
        """
        super().__init__()
        if num_classes < 2 or num_rois < 4 or min_size <= 0 or max_size < min_size:
            raise ValueError("需要至少背景+1类、num_rois>=4，且0<min_size<=max_size。")

        self.backbone = BackboneWithFPN(weights=weights)
        self.rpn = RPNModule()
        self.roi_sampler = RoISampler(num_rois, score_thresh, nms_thresh, detections_per_img)
        self.roi_heads = RoIHeads(num_classes=num_classes)

        self.num_classes = num_classes
        self.min_size = min_size
        self.max_size = max_size
        self.num_rois = num_rois
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def forward(self, images, targets=None, compute_loss=None):
        """
        images: list[float32 Tensor[3,H,W]]，RGB [0,1]，不是已经normalize的batch。
        targets: list[dict]或None，boxes float32[N,4]、labels int64[N]、masks uint8[N,H,W]。
        compute_loss: None跟随train/eval；验证loss显式传True；预测传False。

        返回：
            算loss时 dict[str,标量Tensor]，共5项，交给外面sum、backward。
            预测时 list[dict]，每图boxes[Di,4]/scores[Di]/labels[Di]/masks[Di,1,H,W]。
        """
        if compute_loss is None:
            compute_loss = self.training
        if compute_loss and targets is None:
            raise ValueError("训练/验证loss需要GT；纯预测请model.eval()并不传targets。")

        # 1. 预处理：缩放image/box/mask、RGB normalize、补零并成BCHW。
        images, targets, image_shapes, original_shapes = prepare_images(
            images, targets, self.min_size, self.max_size
        )

        # 2. 一批图片只提取一次共享特征，RPN在P2~P6生成候选框。
        pyramid = self.backbone(images)
        proposals, rpn_losses = self.rpn(
            images, pyramid, targets, image_shapes, compute_loss=compute_loss
        )

        if compute_loss:
            # 3. 用GT给proposal标记、采样；mask还要记住匹配的GT实例下标。
            sampled_proposals, matched_idxs, labels, regression_targets = self.roi_sampler(
                proposals, targets,
            )
            class_logits, box_regression = self.roi_heads.forward_box(
                pyramid, sampled_proposals, image_shapes
            )

            positive_proposals = []
            positive_labels = []
            positive_matched_idxs = []
            for boxes_i, labels_i, matched_i in zip(sampled_proposals, labels, matched_idxs):
                positive = torch.where(labels_i > 0)[0]
                positive_proposals.append(boxes_i[positive])
                positive_labels.append(labels_i[positive])
                positive_matched_idxs.append(matched_i[positive])
            mask_logits = self.roi_heads.forward_mask(
                pyramid, positive_proposals, image_shapes
            )

            # 4. 两项RPN loss + 分类/回归/mask三项loss。这里不做backward/step。
            losses = dict(rpn_losses)
            losses.update(box_loss(
                class_logits, box_regression, labels, regression_targets
            ))
            losses["loss_mask"] = mask_loss(
                mask_logits, positive_proposals, positive_labels,
                positive_matched_idxs, targets,
            )
            return losses

        # 5. 预测没有GT：先分类、修框、阈值过滤、按类别NMS；然后才跑mask。
        class_logits, box_regression = self.roi_heads.forward_box(pyramid, proposals, image_shapes)
        results = postprocess_boxes(
            class_logits, box_regression, proposals, image_shapes,
            self.roi_sampler,
        )
        detection_boxes = [result["boxes"] for result in results]
        mask_logits = self.roi_heads.forward_mask(pyramid, detection_boxes, image_shapes)

        # 6. 框回到原图坐标，28x28局部mask贴回原图，方便直接画图。
        return restore_predictions(results, mask_logits, image_shapes, original_shapes)
