import torch
import torch.nn as nn
from .anchor import getFirstproposal
from .backbone import BackBone
from .fpn import GetPx
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads


class Mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BackBone()  # 采用resnet的
        self.fpn = GetPx()  # 得到level
        self.firstproposalandloss = getFirstproposal()  # RPNs level
        self.roi_heads = RoIHeads(
            box_roi_pool=None,
            box_head=None,
            box_predictor=None,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=64,
            positive_fraction=0.25,
            bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
        )  # 此函数来IOU+sampling，IOU与GTbox大于0.5的作为fore，否则就是back，返回label【】，筛选出来的proposal【【】【】】，匹配的GTbox
        # 检测分支：每个 proposal 提取成 7×7。
        self.box_roi_align = MultiScaleRoIAlign(
            featmap_names=["c1", "c2", "c3", "c4"],
            output_size=7,
            sampling_ratio=2,
        )
        self.branchOp1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self.cls = nn.Linear(1024, 21)
        self.bbox = nn.Linear(1024, 21 * 4)
        # Mask 分支：每个前景 proposal 提取成 14×14。
        self.mask_roi_align = MultiScaleRoIAlign(
            featmap_names=["c1", "c2", "c3", "c4"],
            output_size=14,
            sampling_ratio=2,
        )
        self.branchOp2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 2, 2),
            nn.ReLU(),
            nn.Conv2d(256, 21, 1),
        )

    def forward(self, imageList, targets=None, Train=True):
        # imageList.tensors：Tensor，[B, 3, Hpad, Wpad]。
        # Backbone 接收 Tensor，不是整个 ImageList 对象。
        C = self.backbone(imageList.tensors)

        # C：字典，包含 Backbone 的四层特征。
        # P：字典，包含 FPN 的五层特征。
        P = self.fpn(C)

        # 根据你之前写的接口，返回一个字典：
        # firstLoss：整个 batch 的 RPN loss 字典。
        # firstproposal：每张图片的 proposal Tensor 组成的列表。
        # Train 控制是否需要监督目标和 loss；model.eval() 控制 BN 等层。
        first_result = self.firstproposalandloss(imageList, P, targets, Train)

        if Train:
            if targets is None:
                raise ValueError("训练时需要传入 targets")

            Secondproposals, matched_idxs, labels, regression_targets = (
                self.roi_heads.select_training_samples(
                    first_result["firstproposal"],
                    targets,
                )
            )

            """
                首先对于得到的firstproposal打label，进行筛选（sampling）
            """
            mask_proposals = []
            mask_matched_idxs = []
            mask_labels = []

            for i in range(len(Secondproposals)):
                fg = labels[i] > 0

                mask_proposals.append(Secondproposals[i][fg])
                mask_matched_idxs.append(matched_idxs[i][fg])
                mask_labels.append(labels[i][fg])

        else:
            # 预测没有 GT，不匹配、不抽样。
            # 直接使用 RPN 已经修过的 proposal。
            Secondproposals = first_result["firstproposal"]

        """
        根据列表位置，知道框来自哪张图片。
        根据每个框的宽高，选择 P2～P5 中的一层。
        将图片上的框坐标映射到对应特征图。
        提取固定大小的特征。
        输出对应的特征图的tensor
        """
        # RPN 输出的 firstproposal 已经 decode 过了。
        # 坐标对应缩放后、填充前的图片，不是特征图坐标。
        branch1features = self.box_roi_align(
            P, Secondproposals, imageList.image_sizes
        )  # 根据框面积选择 level；RoIAlign 用双线性插值采样，不像 RoIPool 那样对坐标取整。

        x = self.branchOp1(branch1features)
        cls_logits = self.cls(x)  # Tensor，[R, 21]，背景加20类物体。
        bbox_deltas = self.bbox(x)  # Tensor，[R, 84]，每类4个偏移量。

        if not Train:
            # 第二次 decode、裁剪、分类分数筛选、按类别 NMS。
            # 使用 roi_heads 自己的 BoxCoder，编码和解码权重保持一致。
            boxes, scores, pred_labels = self.roi_heads.postprocess_detections(
                cls_logits,
                bbox_deltas,
                Secondproposals,
                imageList.image_sizes,
            )

            # 预测时，Mask 分支接收最终检测框。
            mask_proposals = boxes

        # 训练：只提取抽样后的前景框特征。
        # 预测：提取第二次修框、NMS 后的最终框特征。
        branch2features = self.mask_roi_align(P, mask_proposals, imageList.image_sizes)

        mask_logits = self.branchOp2(branch2features)
        # 训练：[F, 21, 28, 28]
        # 预测：[D, 21, 28, 28]，D 是整个 batch 的最终检测框总数。

        if Train:
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

        else:
            return {
                "boxes": boxes,  # list[Tensor]，每张图 [Di, 4]
                "scores": scores,  # list[Tensor]，每张图 [Di]
                "labels": pred_labels,  # list[Tensor]，每张图 [Di]
                "mask_logits": mask_logits,  # Tensor，[D, 21, 28, 28]
            }
