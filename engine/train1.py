import torch


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_images = 0
    details = {"rpn_cls": 0.0, "rpn_bbox": 0.0, "cls": 0.0, "bbox": 0.0, "mask": 0.0}

    for imageList, targets in data_loader:
        # 图片和 GT 都放到与模型相同的设备。
        imageList = imageList.to(device)

        for target in targets:
            for key in target:
                target[key] = target[key].to(device)

        # 1. 前向：RPN、检测分支、Mask 分支。
        result = model(imageList, targets, Train=True)

        # 2. 计算检测分类、框回归、Mask loss。
        losses = criterion(result, targets)
        rpn_losses = result["firstLoss"]

        # 3. 五项 loss 相加。
        loss = (
            rpn_losses["loss_objectness"]
            + rpn_losses["loss_rpn_box_reg"]
            + losses["loss_cls"]
            + losses["loss_bbox"]
            + losses["loss_mask"]
        )
        if not torch.isfinite(loss):
            raise ValueError("训练 loss 出现 NaN 或 Inf")

        # 4. 清梯度 → 反向传播 → 更新参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 只在记录数字时使用 item()。
        batch_size = len(targets)
        total_loss += loss.item() * batch_size
        num_images += batch_size
        batch_losses = {
            "rpn_cls": rpn_losses["loss_objectness"],
            "rpn_bbox": rpn_losses["loss_rpn_box_reg"],
            "cls": losses["loss_cls"],
            "bbox": losses["loss_bbox"],
            "mask": losses["loss_mask"],
        }
        for key in details:
            details[key] += batch_losses[key].item() * batch_size

    if num_images == 0:
        raise ValueError("训练集为空")
    for key in details:
        details[key] /= num_images
    total_loss /= num_images
    print(
        f"[Train] 图片={num_images}, loss={total_loss:.4f} | "
        f"RPN分类={details['rpn_cls']:.4f}, RPN回归={details['rpn_bbox']:.4f}, "
        f"RoI分类={details['cls']:.4f}, RoI回归={details['bbox']:.4f}, Mask={details['mask']:.4f}",
        flush=True,
    )
    return total_loss
