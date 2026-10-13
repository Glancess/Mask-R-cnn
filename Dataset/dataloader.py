import torch

from torch.utils.data import DataLoader
from torchvision.models.detection.image_list import ImageList
from torchvision.transforms import functional as F

from .dataset import PennFudanDataset


def collate_fn(batch):
    """
    输入：
        batch：list，每个元素是 Dataset 返回的 (image, target)。

    返回：
        imageList：包含补齐后的图片，以及每张图片补齐前的大小。
        targets：list[dict]，每张图片的 GT。
    """
    images = []
    targets = []
    image_sizes = []

    for image, target in batch:
        # image：[3,H,W]。
        h, w = image.shape[-2:]
        image_sizes.append((h, w))

        # RGB 归一化，不改变图片大小，也不改变 GT。
        image = F.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        images.append(image)
        targets.append(target)

    # 找到这个 batch 中最大的高和宽。
    max_h = max(size[0] for size in image_sizes)
    max_w = max(size[1] for size in image_sizes)

    # 补到 64 的整数倍，方便生成 P2～P6。
    max_h = ((max_h + 63) // 64) * 64
    max_w = ((max_w + 63) // 64) * 64

    # 创建整批图片的存放空间。
    batch_images = images[0].new_zeros((len(images), 3, max_h, max_w))

    for i in range(len(images)):
        h, w = image_sizes[i]

        # 图片放左上角，右边和下面保留补零区域。
        batch_images[i, :, :h, :w] = images[i]

    imageList = ImageList(batch_images, image_sizes)

    return imageList, targets


def get_dataloader(root, batch_size=2, shuffle=True, num_workers=0):
    """
    root：数据集路径，内部包含 PNGImages、PedMasks、Annotation。
    返回：DataLoader。
    """
    dataset = PennFudanDataset(root)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return data_loader


if __name__ == "__main__":
    data_loader = get_dataloader(
        root="/Users/alen/Code/Pytorcc/projects/Mask-R-cnn/data/PennFudanPed",
        batch_size=2,
    )

    imageList, targets = next(iter(data_loader))

    print("batch 图片：", imageList.tensors.shape)
    print("每张图有效大小：", imageList.image_sizes)

    for target in targets:
        print("boxes：", target["boxes"].shape)
        print("labels：", target["labels"].shape)
        print("masks：", target["masks"].shape)
