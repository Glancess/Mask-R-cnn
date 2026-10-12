def collate_fn(batch):
    """
    输入 batch: list[tuple(image, target)]，来自 Dataset.__getitem__。
    返回 (images, targets)：两个长度为 B 的 list。
        images[i]: float32 Tensor[3,Hi,Wi]
        targets[i]: dict，包含 boxes / labels / masks

    图片尺寸、行人数不同，不能直接 stack；先保留列表，交给模型补零。
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets
