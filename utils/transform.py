import torch
import torch.nn.functional as F


def pad_images(images):
    """
    输入 images: list[float32 Tensor[3,Hi,Wi]]，已经缩放、归一化。
    返回 (batch_images, image_shapes)：
        batch_images: Tensor[B,3,H_pad,W_pad]，右侧和下侧补零。
        image_shapes: list[(Hi,Wi)]，补零前的真实尺寸。
    从 engine 移到这里，只为让 train / val / predict 共用同一份预处理。
    """
    if len(images) == 0:
        raise ValueError("images 不能为空。")

    image_shapes = [(img.shape[-2], img.shape[-1]) for img in images]
    max_h = max(h for h, w in image_shapes)
    max_w = max(w for h, w in image_shapes)

    # P6 的 stride=64，补到64的倍数，AnchorGenerator 才能算出准确的 stride。
    max_h = (max_h + 63) // 64 * 64
    max_w = (max_w + 63) // 64 * 64
    batch_images = images[0].new_zeros((len(images), 3, max_h, max_w))

    for i, image in enumerate(images):
        h, w = image_shapes[i]
        batch_images[i, :, :h, :w] = image

    return batch_images, image_shapes


def prepare_images(images, targets=None, min_size=400, max_size=640):
    """
    输入 images: list[float32 Tensor[3,H,W]]，RGB [0,1]，不要提前 normalize。
    targets: list[dict]或None，包含 boxes[N,4] / labels[N] / masks[N,H,W]。
    min_size / max_size: int，期望短边和最长边上限，保持图片长宽比。

    返回四项：
        batch_images: float32 Tensor[B,3,H_pad,W_pad]
        resized_targets: list[dict]或None，与缩放后的图片坐标对应
        image_shapes: list[(H_new,W_new)]，用于 RPN / RoIAlign
        original_shapes: list[(H,W)]，用于把预测还原到原图

    不原地修改 Dataset 返回的 image / target，画图仍能使用原图和原 GT。
    """
    if len(images) == 0 or min_size <= 0 or max_size < min_size:
        raise ValueError("需要非空 images，且 0 < min_size <= max_size。")
    if targets is not None and len(images) != len(targets):
        raise ValueError("每张图片必须有一个对应的 target。")

    resized_images = []
    resized_targets = [] if targets is not None else None
    original_shapes = []

    for i, image in enumerate(images):
        if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
            raise ValueError("图片必须是 RGB float Tensor[3,H,W]，范围[0,1]。")

        h, w = image.shape[-2:]
        original_shapes.append((h, w))

        # 1. 保持比例缩放。image 用双线性插值，mask 只能用最近邻插值。
        scale = min(min_size / min(h, w), max_size / max(h, w))
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        resized_image = F.interpolate(
            image[None], size=(new_h, new_w), mode="bilinear", align_corners=False,
        )[0]

        # 2. 与 ImageNet 预训练 ResNet 一致的 RGB mean / std。
        mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        resized_images.append((resized_image - mean) / std)

        if targets is not None:
            target = dict(targets[i])
            # 使用实际整数尺寸的比例，而不是取整前的 scale，避免框与图错位。
            ratio = image.new_tensor([new_w / w, new_h / h, new_w / w, new_h / h])
            target["boxes"] = target["boxes"] * ratio
            masks = target["masks"]
            if len(masks) == 0:
                target["masks"] = masks.new_empty((0, new_h, new_w))
            else:
                target["masks"] = F.interpolate(
                    masks[:, None].float(), size=(new_h, new_w), mode="nearest",
                )[:, 0].to(masks.dtype)
            resized_targets.append(target)

    batch_images, image_shapes = pad_images(resized_images)
    return batch_images, resized_targets, image_shapes, original_shapes
