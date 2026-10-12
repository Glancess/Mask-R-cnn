import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image


def draw_instances(image, boxes, labels, masks, scores=None, mask_thresh=0.5):
    """
    image: float32[3,H,W]，原始 RGB [0,1]。
    boxes: float32[N,4]；labels: int64[N]；masks: [N,H,W] 或 [N,1,H,W]。
    scores: float32[N]或None；mask_thresh: float，把预测概率变成二值mask。
    返回 PIL.Image：同时画检测框和半透明实例mask，每个实例使用不同颜色。
    """
    canvas = to_pil_image(image.detach().cpu()).convert("RGBA")
    boxes = boxes.detach().cpu()
    labels = labels.detach().cpu()
    masks = masks.detach().cpu()
    if masks.ndim == 4:
        masks = masks[:, 0]
    if scores is not None:
        scores = scores.detach().cpu()

    colors = [(255, 80, 80), (40, 200, 80), (60, 140, 255),
              (255, 180, 40), (200, 80, 240), (40, 210, 210)]

    # 1. 先叠加各个实例的 mask。不是把所有 person 合并成一种颜色。
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        alpha = (mask.numpy() > mask_thresh).astype(np.uint8) * 90
        layer = Image.new("RGBA", canvas.size, color + (0,))
        layer.putalpha(Image.fromarray(alpha))
        canvas = Image.alpha_composite(canvas, layer)

    # 2. 再画框和标签，避免被后画的 mask 遮住。
    draw = ImageDraw.Draw(canvas)
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = "person" if labels[i].item() == 1 else str(labels[i].item())
        if scores is not None:
            text += f" {scores[i].item():.2f}"
        draw.text((x1, max(0, y1 - 12)), text, fill=color)

    return canvas.convert("RGB")


def save_comparison(image, target, prediction, save_path):
    """
    输入原图 Tensor、原始GT dict、预测dict、保存路径 str。
    返回 str：图片路径。左边GT，右边Prediction，双方都画框+实例mask。
    """
    gt_image = draw_instances(image, target["boxes"], target["labels"], target["masks"])
    pred_image = draw_instances(
        image, prediction["boxes"], prediction["labels"], prediction["masks"],
        scores=prediction["scores"],
    )
    w, h = gt_image.size
    canvas = Image.new("RGB", (w * 2, h + 24), "white")
    canvas.paste(gt_image, (0, 24))
    canvas.paste(pred_image, (w, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), "Ground Truth: boxes + instance masks", fill="black")
    draw.text((w + 5, 5), "Prediction: boxes + instance masks", fill="black")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    canvas.save(save_path)
    return save_path


@torch.no_grad()
def test_best_model(model, dataset, test_indices, save_dir, device, num_images=None):
    """
    model: 已加载best.pth的MaskRCNN；dataset: PennFudanDataset。
    test_indices: list[int]，必须来自训练时保存的 test 划分。
    save_dir: str；device: torch.device；num_images: int或None（全部）。
    返回 list[str]：生成的对比图路径。只看最终预测，不计算 AP/mAP。
    """
    model.eval()
    if num_images is not None:
        if num_images <= 0:
            raise ValueError("num_images 应为正整数或 None。")
        test_indices = test_indices[:num_images]

    saved_paths = []
    for index in test_indices:
        image, target = dataset[index]

        # GT 不传给模型！只在预测完成后交给画图函数做左侧对照。
        prediction = model([image.to(device)], compute_loss=False)[0]
        prediction = {key: value.cpu() for key, value in prediction.items()}
        stem = os.path.splitext(dataset.imgs[index])[0]
        save_path = os.path.join(save_dir, stem + "_compare.png")
        saved_paths.append(save_comparison(image, target, prediction, save_path))
        print(f"test {dataset.imgs[index]}: 预测 {len(prediction['boxes'])} 个实例 -> {save_path}")
    return saved_paths
