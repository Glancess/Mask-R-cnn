import os

import torch
from PIL import Image, ImageDraw
from torchvision.models.detection.roi_heads import (
    maskrcnn_inference,
    paste_masks_in_image,
)

from Model.mask import Mask
from Dataset.dataloader import getVOC_dataloader, collate_fn
from engine.test import draw_result


@torch.no_grad()
def main():
    root = os.path.dirname(os.path.abspath(__file__))
    best_path = os.path.join(root, "best_voc.pth")
    save_dir = os.path.join(root, "testimage")
    num_images = 10
    score_thresh = 0.5

    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"找不到最佳权重：{best_path}")

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 沿用训练时的seed=0划分，只取val；val不随机翻转。
    _, val_loader, _ = getVOC_dataloader(os.path.join(root, "data"))
    val_dataset = val_loader.dataset

    model = Mask().to(device)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    print(f"加载：{best_path} | 设备：{device} | 分数阈值：{score_thresh}", flush=True)

    # 只预测验证集前10张，不遍历全部val，也不重新训练。
    for i in range(num_images):
        image, target = val_dataset[i]  # 原图[3,H,W]，GT字典。
        h, w = image.shape[-2:]
        imageList, _ = collate_fn([(image, target)])

        # 纯预测不传GT；GT只用于左侧对照图。
        result = model(imageList.to(device), Train=False)
        boxes = result["boxes"][0]  # [D,4]，最终修框和NMS后的结果。
        scores = result["scores"][0]  # [D]。
        labels = result["labels"][0]  # [D]。
        masks = maskrcnn_inference(result["mask_logits"], result["labels"])[0]
        # masks：[D,1,28,28]，已选预测类别通道并sigmoid。

        keep = scores >= score_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        masks = masks[keep]

        # 将小mask放回图片；也允许这张图没有超过阈值的预测。
        if len(boxes) > 0:
            masks = paste_masks_in_image(masks, boxes, (h, w))[:, 0] >= 0.5
        else:
            masks = torch.empty((0, h, w), dtype=torch.bool, device=device)

        gt = draw_result(image, target["boxes"], target["masks"], target["labels"])
        pred = draw_result(image, boxes.cpu(), masks.cpu(), labels.cpu(), scores.cpu())

        comparison = Image.new("RGB", (w * 2, h + 24), "white")
        comparison.paste(gt, (0, 24))
        comparison.paste(pred, (w, 24))
        draw = ImageDraw.Draw(comparison)
        draw.text((5, 5), "Ground Truth", fill="black")
        draw.text((w + 5, 5), "Prediction", fill="black")

        # val_dataset是Subset，用其下标找到原数据集中的图片名。
        index = val_dataset.indices[i]
        image_id = os.path.splitext(val_dataset.dataset.imgs[index])[0]
        name = f"val_{i + 1:02d}_{image_id}_compare.png"
        comparison.save(os.path.join(save_dir, name))
        print(
            f"[Val {i + 1}/{num_images}] {image_id} | "
            f"GT目标={len(target['boxes'])}, 预测目标={len(boxes)} | {name}",
            flush=True,
        )

    print(f"完成：{num_images}张对比图已保存到 {save_dir}", flush=True)


if __name__ == "__main__":
    main()
