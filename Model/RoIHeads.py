import torch
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign


class RoIHeads(nn.Module):
    def __init__(self, num_classes=2):
        """
        num_classes:
            int，包含 background。

            PennFudan:
                background = 0
                person     = 1

            所以 num_classes = 2
        """
        super().__init__()

        self.num_classes = num_classes

        # ==================================================
        # 1. Box / Classification RoIAlign
        #
        # proposal
        #   ↓
        # 根据 proposal 尺寸自动选择 P2/P3/P4/P5
        #   ↓
        # RoIAlign
        #   ↓
        # [N, 256, 7, 7]
        # ==================================================
        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["p2", "p3", "p4", "p5"],
            output_size=7,
            sampling_ratio=2,
        )

        # ==================================================
        # 2. Fast R-CNN Box Head
        #
        # [N,256,7,7]
        # ↓ flatten
        # [N, 256*7*7]
        # ↓
        # FC 1024
        # ↓
        # FC 1024
        # ==================================================
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )

        # classification
        self.cls_score = nn.Linear(1024, num_classes)

        # class-specific bbox regression
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

        # ==================================================
        # 3. Mask RoIAlign
        #
        # positive proposal
        # ↓
        # 根据 proposal 尺寸选择 P2/P3/P4/P5
        # ↓
        # RoIAlign
        # ↓
        # [N_pos,256,14,14]
        # ==================================================
        self.mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=["p2", "p3", "p4", "p5"],
            output_size=14,
            sampling_ratio=2,
        )

        # ==================================================
        # 4. Mask FCN Head
        #
        # [N_pos,256,14,14]
        # ↓
        # 4 x 3x3 Conv
        # ↓
        # [N_pos,256,14,14]
        # ↓
        # Deconv x2
        # ↓
        # [N_pos,256,28,28]
        # ==================================================
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        # 每个类别一张 mask
        self.mask_predictor = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward_box(self, pyramid, proposals, image_shapes):
        """
        输入 pyramid: dict[str,Tensor]；proposals: list[float32[Ni,4]]；
        image_shapes: list[(H,W)]，真实尺寸。返回 tuple：
        class_logits float32[R,C]、box_regression float32[R,C*4]。
        R=sum(Ni)，C包含背景；这里只跑 box 分支，不做分类筛选或 NMS。
        """
        # MultiScaleRoIAlign 用尺寸推断 stride。特征图来自 padding 后的图片，
        # 所以这里用 padding 尺寸推断；proposal 本身仍是未补零的图片坐标。
        # P2 stride=4，这样小尺寸 smoke test 也不会因补零过多推错 stride。
        padded_shape = (pyramid["p2"].shape[-2] * 4, pyramid["p2"].shape[-1] * 4)
        pool_shapes = [padded_shape] * len(image_shapes)
        box_features = self.box_roi_pool(pyramid, proposals, pool_shapes)
        box_features = self.box_head(box_features)
        class_logits = self.cls_score(box_features)
        box_regression = self.bbox_pred(box_features)

        return class_logits, box_regression

    def forward_mask(self, pyramid, boxes, image_shapes):
        """
        输入 boxes: list[float32[Ni,4]]；其余与 forward_box 相同。
        训练传正样本 proposal；预测传经过修框、筛分、NMS 的最终框。
        返回 mask_logits: float32[sum(Ni),C,28,28]，还没有 sigmoid。
        """
        padded_shape = (pyramid["p2"].shape[-2] * 4, pyramid["p2"].shape[-1] * 4)
        pool_shapes = [padded_shape] * len(image_shapes)
        mask_features = self.mask_roi_pool(pyramid, boxes, pool_shapes)
        mask_features = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_features)

        return mask_logits

    def forward(
        self,
        pyramid,
        proposals,
        image_shapes,
        positive_proposals=None,
    ):
        """
        pyramid:
            dict[str, Tensor]，每项都是float32特征图。
            {
                "p2": [B,256,H2,W2],
                "p3": [B,256,H3,W3],
                "p4": [B,256,H4,W4],
                "p5": [B,256,H5,W5],
                "p6": [B,256,H6,W6],
            }

        proposals:
            list[Tensor]，长度B，框使用当前图片坐标系的xyxy，不要手动除stride。
            proposals[i] = 第 i 张图片的 RoIs
            shape: [Ni,4]

        image_shapes:
            list[tuple[int,int]]，真实(H,W)，不是补零后的统一尺寸。
            [
                (H1,W1),
                (H2,W2),
                ...
            ]

        positive_proposals:
            list[Tensor]或None，长度B，每项float32 Tensor[N_pos_i,4]。
            训练时 mask branch 只使用 positive RoIs；列表位置不能删除或错位。

            必须显式传入；没有正样本时也保留每张图的空 Tensor[0,4]。

        返回:
            dict[str, Tensor]，三项都是原始输出，还没有softmax/sigmoid：
                class_logits:   [R,num_classes]，R=sum(Ni)
                box_regression: [R,num_classes*4]，包含背景的回归通道
                mask_logits:    [R_pos,num_classes,28,28]
            R_pos是mask分支输入框的总数，不一定等于R。

        注意:
            这个类只做网络前向；GT匹配、loss和NMS由Mask.py和utils负责。
            正式推理应该先得到最终检测框，再用这些框跑mask分支。
            mask_logits不是整张图的mask，28x28是各个RoI内的局部坐标。
        """

        # ==================================================
        # Box branch
        # ==================================================

        class_logits, box_regression = self.forward_box(
            pyramid, proposals, image_shapes
        )

        # ==================================================
        # Mask branch
        # ==================================================

        # 正式训练时这里必须只传 positive RoIs。
        if positive_proposals is None:
            raise ValueError("Training mask branch requires positive_proposals.")

        mask_logits = self.forward_mask(pyramid, positive_proposals, image_shapes)

        return {
            "class_logits": class_logits,
            "box_regression": box_regression,
            "mask_logits": mask_logits,
        }
