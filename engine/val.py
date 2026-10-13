import torch


@torch.no_grad()
def val(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_images = 0
    details = {"rpn_cls": 0.0, "rpn_bbox": 0.0, "cls": 0.0, "bbox": 0.0, "mask": 0.0}

    for imageList, targets in data_loader:
        imageList = imageList.to(device)
        for target in targets:
            for key in target:
                target[key] = target[key].to(device)

        # Train=True 只用于取得 loss 所需的监督结果，不执行反向传播。
        # model.eval() 仍然保证 BN 不更新。
        result = model(imageList, targets, Train=True)
        losses = criterion(result, targets)
        rpn_losses = result["firstLoss"]

        loss = (
            rpn_losses["loss_objectness"]
            + rpn_losses["loss_rpn_box_reg"]
            + losses["loss_cls"]
            + losses["loss_bbox"]
            + losses["loss_mask"]
        )
        if not torch.isfinite(loss):
            raise ValueError("验证 loss 出现 NaN 或 Inf")

        total_loss += loss.item() * len(targets)
        num_images += len(targets)
        batch_losses = {
            "rpn_cls": rpn_losses["loss_objectness"],
            "rpn_bbox": rpn_losses["loss_rpn_box_reg"],
            "cls": losses["loss_cls"],
            "bbox": losses["loss_bbox"],
            "mask": losses["loss_mask"],
        }
        for key in details:
            details[key] += batch_losses[key].item() * len(targets)

    if num_images == 0:
        raise ValueError("验证集为空")
    for key in details:
        details[key] /= num_images
    total_loss /= num_images
    print(
        f"[Val]   图片={num_images}, loss={total_loss:.4f} | "
        f"RPN分类={details['rpn_cls']:.4f}, RPN回归={details['rpn_bbox']:.4f}, "
        f"RoI分类={details['cls']:.4f}, RoI回归={details['bbox']:.4f}, Mask={details['mask']:.4f}",
        flush=True,
    )
    return total_loss
