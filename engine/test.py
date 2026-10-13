import os

import torch
from PIL import Image, ImageDraw
from torchvision.models.detection.roi_heads import (
    maskrcnn_inference,
    paste_masks_in_image,
)
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks



def draw_result(image, boxes, masks, scores=None):
    """输入原图 [3,H,W]、框 [N,4]、mask [N,H,W]；返回带标注的 PIL 图片。"""
    image = (image * 255).clamp(0, 255).to(torch.uint8)
    colors = ["red", "green", "blue", "orange", "purple", "cyan"]
    colors = [colors[i % len(colors)] for i in range(len(boxes))]
    if len(boxes) > 0:
        image = draw_segmentation_masks(
            image, masks.bool(), alpha=0.4, colors=colors
        )
        if scores is None:
            labels = ["person"] * len(boxes)
        else:
            labels = [f"person {score:.2f}" for score in scores.tolist()]
        image = draw_bounding_boxes(
            image, boxes, labels=labels, colors=colors, width=2
        )
    return F.to_pil_image(image)


@torch.no_grad()
def test_best_model(model, test_loader, device, save_dir, score_thresh=0.5, num_images=10):
    """测试全部图片，打印预测概况；按 loader 顺序保存前 10 张对比图。"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    total_gt = 0
    total_pred = 0
    total_images = 0
    saved_images = 0
    print(f"[Test] 分数阈值={score_thresh}，开始测试", flush=True)

    for i, (imageList, targets) in enumerate(test_loader):
        h, w = imageList.image_sizes[0]
        # 当前 loader 没有缩放，因此有效图片大小就是原图大小。
        image = imageList.tensors[0, :, :h, :w] * std + mean
        image = image.clamp(0, 1)

        # GT 只用于画左边的对照图，不能传给预测模型。
        result = model(imageList.to(device), Train=False)
        boxes = result["boxes"][0]
        scores = result["scores"][0]

        # 按预测类别选 Mask 通道并 sigmoid，得到 [D,1,28,28]。
        masks = maskrcnn_inference(result["mask_logits"], result["labels"])[0]
        keep = scores >= score_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        masks = masks[keep]

        # test_loader 不打乱顺序；找到这张图片的名字。
        test_dataset = test_loader.dataset
        index = test_dataset.indices[i]
        name = os.path.splitext(test_dataset.dataset.imgs[index])[0]
        gt_count = len(targets[0]["boxes"])
        total_gt += gt_count
        total_pred += len(boxes)
        total_images += 1
        score_text = [round(x, 3) for x in scores[:5].tolist()]
        print(
            f"[Test {i + 1}/{len(test_loader)}] {name} | "
            f"GT行人={gt_count}, 预测行人={len(boxes)}, 前5个分数={score_text}",
            flush=True,
        )

        # 全部 test 都预测，但只保存前 10 张，不按预测好坏挑图片。
        if i >= num_images:
            continue

        # 把每个 28×28 mask 放回最终检测框，成为整图 mask。
        if len(boxes) > 0:
            masks = paste_masks_in_image(masks, boxes, (h, w))[:, 0] >= 0.5
        else:
            masks = torch.empty((0, h, w), dtype=torch.bool, device=device)

        gt = draw_result(image, targets[0]["boxes"], targets[0]["masks"])
        pred = draw_result(image, boxes.cpu(), masks.cpu(), scores.cpu())

        # 左边 GT，右边预测；相同实例的框和 mask 使用相同颜色。
        comparison = Image.new("RGB", (w * 2, h + 24), "white")
        comparison.paste(gt, (0, 24))
        comparison.paste(pred, (w, 24))
        draw = ImageDraw.Draw(comparison)
        draw.text((5, 5), "Ground Truth", fill="black")
        draw.text((w + 5, 5), "Prediction", fill="black")
        comparison.save(os.path.join(save_dir, name + "_compare.png"))
        saved_images += 1

    print(
        f"[Test完成] 图片={total_images}, GT行人总数={total_gt}, "
        f"预测行人总数={total_pred}, 保存对比图={saved_images}",
        flush=True,
    )
    print(f"对比图目录：{save_dir}（左GT，右预测，含框和实例mask）", flush=True)
    print("以上为预测数量和分数，不是准确率或 mAP。", flush=True)
