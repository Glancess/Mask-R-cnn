import torch
from utils.transform import pad_images, prepare_images


def freeze_bn_stats(backbone):
    """
    输入 backbone: nn.Module；无返回值。
    batch=2 太小，只固定 ResNet BatchNorm 的 running mean/var。
    这不是冻结 backbone：卷积和 BN 的 weight / bias 仍然能求梯度、更新。
    """
    for module in backbone.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


def get_positive_samples(
    sampled_proposals,
    labels,
    matched_idxs,
):
    """
    Mask branch 只处理 positive RoI。

    输入三个长度 B 的 list：
        sampled_proposals[i]: float32[Ri,4]，xyxy
        labels[i]: int64[Ri]，0为背景，>0为前景
        matched_idxs[i]: int64[Ri]，本图GT实例下标

    返回:
        positive_proposals: list[float32[Pi,4]]
        positive_labels: list[int64[Pi]]
        positive_matched_idxs: list[int64[Pi]]

    Pi是第i张图的正样本数；三者索引完全对应，空图片位置也不能删。
    """

    positive_proposals = []
    positive_labels = []
    positive_matched_idxs = []

    for proposals_i, labels_i, matched_i in zip(
        sampled_proposals,
        labels,
        matched_idxs,
    ):
        positive_idx = torch.where(labels_i > 0)[0]

        positive_proposals.append(proposals_i[positive_idx])

        positive_labels.append(labels_i[positive_idx])

        positive_matched_idxs.append(matched_i[positive_idx])

    return (
        positive_proposals,
        positive_labels,
        positive_matched_idxs,
    )


def train_one_epoch(
    backbone,
    rpn,
    roi_sampler,
    roi_heads,
    dataloader,
    optimizer,
    device,
    box_loss_fn,
    mask_loss_fn,
    epoch=0,
    min_size=400,
    max_size=640,
):
    """
    backbone:
        BackboneWithFPN

    rpn:
        RPNModule

    roi_sampler:
        你封装的 RoI sampling 黑盒

    roi_heads:
        你自己的 RoIHeads
        包含：
            forward_box()
            forward_mask()

    dataloader:
        每个 batch 返回：
            images, targets

        推荐 collate_fn 让 images / targets 都保持 list。

    optimizer:
        torch optimizer

    device:
        cuda / mps / cpu

    box_loss_fn:
        你的 box_loss()

    mask_loss_fn:
        你的 mask_loss()

    min_size / max_size:
        int，train / val / predict 必须使用同一组预处理尺寸。

    返回:
        float，这一轮的平均总 loss，仍然保留你原来的返回类型。
    """

    # =====================================================
    # 进入 train mode
    # =====================================================

    backbone.train()
    freeze_bn_stats(backbone)
    rpn.train()
    roi_heads.train()

    total_epoch_loss = 0.0
    total_images = 0

    for batch_idx, (images, targets) in enumerate(dataloader):

        # =================================================
        # 1. 数据放到 device
        #
        # images:
        # [
        #   Tensor[3,H1,W1],
        #   Tensor[3,H2,W2],
        #   ...
        # ]
        #
        # targets:
        # [
        #   {
        #       boxes:  [N1,4],
        #       labels: [N1],
        #       masks:  [N1,H1,W1]
        #   },
        #   ...
        # ]
        # =================================================

        images = [image.to(device) for image in images]

        targets = [
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()}
            for target in targets
        ]

        # =================================================
        # 2. 同步缩放 image/boxes/masks，RGB normalize，再 padding 成 batch
        #
        # 原尺寸必须保留
        # =================================================

        batch_images, targets, image_shapes, _ = prepare_images(
            images, targets, min_size, max_size,
        )

        # batch_images:
        # [B,3,Hmax,Wmax]

        # image_shapes:
        # [(H1,W1), (H2,W2), ...]

        # =================================================
        # 3. Backbone + FPN
        #
        # image
        # ↓
        # ResNet
        # ↓
        # C2~C5
        # ↓
        # FPN
        # ↓
        # P2~P6
        # =================================================

        pyramid = backbone(batch_images)

        # pyramid:
        #
        # {
        #   p2: [B,256,...],
        #   p3: [B,256,...],
        #   p4: [B,256,...],
        #   p5: [B,256,...],
        #   p6: [B,256,...],
        # }

        # =================================================
        # 4. 第一阶段 RPN
        #
        # RPN 内部完成：
        #
        # anchor generation
        # anchor ↔ GT matching
        # anchor sampling
        # objectness loss
        # bbox loss
        #
        # 同时：
        #
        # anchor
        # + predicted delta
        # ↓
        # proposal
        # ↓
        # TopK + NMS
        # =================================================

        proposals, rpn_losses = rpn(
            images=batch_images,
            pyramid=pyramid,
            targets=targets,
            image_sizes=image_shapes,
        )

        # proposals:
        # [
        #   Tensor[N1,4],
        #   Tensor[N2,4],
        #   ...
        # ]
        #
        # 已经是 RPN 第一次修过的框
        #
        # rpn_losses:
        # {
        #   loss_objectness,
        #   loss_rpn_box_reg
        # }

        # =================================================
        # 5. 第二阶段 RoI Sampling
        #
        # proposal ↔ GT
        # ↓
        # IoU
        # ↓
        # class label
        # ↓
        # sampling
        #
        # 黑盒全部完成
        # =================================================

        (
            sampled_proposals,
            matched_idxs,
            labels,
            regression_targets,
        ) = roi_sampler(
            proposals,
            targets,
        )

        # sampled_proposals:
        #   第二阶段真正拿去训练的 RoIs
        #
        # labels:
        #   0 = background
        #   1 = person
        #
        # matched_idxs:
        #   每个 RoI 对应哪个 GT instance
        #
        # regression_targets:
        #   proposal -> GT 的真实 bbox delta

        # =================================================
        # 6. Box / Classification Branch
        #
        # sampled proposal
        # ↓
        # 根据尺度选择 P2~P5
        # ↓
        # RoIAlign 7x7
        # ↓
        # FC
        # ↓
        # cls + bbox
        # =================================================

        (
            class_logits,
            box_regression,
        ) = roi_heads.forward_box(
            pyramid,
            sampled_proposals,
            image_shapes,
        )

        # class_logits:
        # [R, num_classes]
        #
        # box_regression:
        # [R, num_classes * 4]

        # =================================================
        # 7. Fast R-CNN Loss
        #
        # class_logits
        # VS
        # labels
        #
        # predicted bbox delta
        # VS
        # regression_targets
        # =================================================

        box_losses = box_loss_fn(
            class_logits,
            box_regression,
            labels,
            regression_targets,
        )

        # {
        #   loss_classifier,
        #   loss_box_reg
        # }

        # =================================================
        # 8. 取 Positive RoIs
        #
        # Mask branch 不处理 background
        # =================================================

        (
            positive_proposals,
            positive_labels,
            positive_matched_idxs,
        ) = get_positive_samples(
            sampled_proposals,
            labels,
            matched_idxs,
        )

        # =================================================
        # 9. Mask Branch
        #
        # positive proposal
        # ↓
        # 选择 P2~P5
        # ↓
        # RoIAlign 14×14
        # ↓
        # FCN
        # ↓
        # 28×28
        # ↓
        # C mask channels
        # =================================================

        mask_logits = roi_heads.forward_mask(
            pyramid,
            positive_proposals,
            image_shapes,
        )

        # [P, num_classes, 28, 28]

        # =================================================
        # 10. Mask Loss
        #
        # matched_idxs
        # ↓
        # 找 matched GT instance mask
        # ↓
        # crop/project → 28×28
        #
        # mask_logits 中只取 GT class 对应通道
        #
        # ↓
        # BCEWithLogits
        # =================================================

        loss_mask = mask_loss_fn(
            mask_logits,
            positive_proposals,
            positive_labels,
            positive_matched_idxs,
            targets,
        )

        # =================================================
        # 11. 总 Loss
        # =================================================

        loss_objectness = rpn_losses["loss_objectness"]

        loss_rpn_box = rpn_losses["loss_rpn_box_reg"]

        loss_classifier = box_losses["loss_classifier"]

        loss_box = box_losses["loss_box_reg"]

        total_loss = (
            loss_objectness + loss_rpn_box + loss_classifier + loss_box + loss_mask
        )
        if not torch.isfinite(total_loss).item():
            raise ValueError(f"训练 loss 出现 NaN/Inf，epoch={epoch}, batch={batch_idx}。")

        # =================================================
        # 12. Backward
        # =================================================

        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()

        # =================================================
        # 13. Logging
        # =================================================

        # 最后一个 batch 可能少一张图，按图片数加权，避免把小 batch 算得过重。
        total_epoch_loss += total_loss.item() * len(images)
        total_images += len(images)

        if batch_idx % 10 == 0:

            num_rois = sum(len(p) for p in sampled_proposals)

            num_positive = sum(len(p) for p in positive_proposals)

            print(
                f"[Epoch {epoch}] "
                f"[Batch {batch_idx}/{len(dataloader)}] "
                f"loss={total_loss.item():.4f} | "
                f"rpn_obj={loss_objectness.item():.4f} | "
                f"rpn_box={loss_rpn_box.item():.4f} | "
                f"cls={loss_classifier.item():.4f} | "
                f"box={loss_box.item():.4f} | "
                f"mask={loss_mask.item():.4f} | "
                f"RoIs={num_rois} | "
                f"Pos={num_positive}"
            )

    if total_images == 0:
        raise ValueError("训练 dataloader 为空，请检查 train 划分。")
    avg_loss = total_epoch_loss / total_images

    return avg_loss


@torch.no_grad()
def evaluate_one_epoch(
    backbone, rpn, roi_sampler, roi_heads, dataloader, device,
    box_loss_fn, mask_loss_fn, min_size=400, max_size=640,
):
    """
    输入模块、DataLoader、device、loss函数，类型与 train_one_epoch 一样。
    返回 float：平均验证总 loss。只算五项 loss，不做修框后的检测评价。
    GT 只用来匹配与监督，没有 backward / optimizer.step，不更新模型。
    """
    backbone.eval()
    rpn.eval()
    roi_heads.eval()
    total_loss = 0.0
    total_images = 0

    # RPN 和 RoISampler 内部都有随机采样。固定验证采样，方便逐轮比较。
    # fork_rng 在验证后恢复原 RNG，验证不会改变下一轮训练的采样序列。
    device = torch.device(device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.default_generators[cuda_devices[0]].manual_seed(0)

        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            targets = [
                {k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()}
                for target in targets
            ]
            batch_images, targets, image_shapes, _ = prepare_images(
                images, targets, min_size, max_size,
            )

            # 1. 和训练一样提取特征、生成 proposal；只让 RPN 返回 loss。
            # backbone 保持 eval，不能为了算 loss 把整套网络改回 train。
            pyramid = backbone(batch_images)
            proposals, rpn_losses = rpn(
                batch_images, pyramid, targets, image_shapes, compute_loss=True,
            )

            # 2. 同样用 GT 匹配、采样，才能与训练时的 loss 做比较。
            sampled_proposals, matched_idxs, labels, regression_targets = roi_sampler(
                proposals, targets,
            )
            class_logits, box_regression = roi_heads.forward_box(
                pyramid, sampled_proposals, image_shapes,
            )
            box_losses = box_loss_fn(class_logits, box_regression, labels, regression_targets)

            # 3. 背景不进 mask branch。列表里每张图的位置仍然保留。
            positive_proposals, positive_labels, positive_matched_idxs = get_positive_samples(
                sampled_proposals, labels, matched_idxs,
            )
            mask_logits = roi_heads.forward_mask(pyramid, positive_proposals, image_shapes)
            loss_mask = mask_loss_fn(
                mask_logits, positive_proposals, positive_labels, positive_matched_idxs, targets,
            )

            loss = (
                rpn_losses["loss_objectness"] + rpn_losses["loss_rpn_box_reg"]
                + box_losses["loss_classifier"] + box_losses["loss_box_reg"] + loss_mask
            )
            if not torch.isfinite(loss).item():
                raise ValueError("验证 loss 出现 NaN/Inf。")
            total_loss += loss.item() * len(images)
            total_images += len(images)

    if total_images == 0:
        raise ValueError("验证 dataloader 为空，请检查 val 划分。")
    return total_loss / total_images
