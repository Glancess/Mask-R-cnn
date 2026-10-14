import torch

from torch.utils.data import DataLoader, random_split
from torchvision.models.detection.image_list import ImageList
from torchvision.transforms import functional as F

# from .dataset import PennFudanDataset
from .Vocdataset import VocDataset


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


# def get_dataloader(root, batch_size=2, shuffle=True, num_workers=0):
#     """
#     root：数据集路径，内部包含 PNGImages、PedMasks、Annotation。
#     返回：DataLoader。
#     """
#     dataset = PennFudanDataset(root)

#     data_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         collate_fn=collate_fn,
#     )

#     return data_loader


def getVOC_dataloader(root, batch_size=2, shuffle=True, num_workers=0):
    """
    root：包含 VOCdevkit/VOC2012 的目录。
    返回：train_loader、val_loader、test_loader，三份图片互不重叠。
    """
    # VOC2012 官方 test 没有公开分割 GT，采用下面的自定义学习划分。
    # 官方 train1464：1264训练、200验证；官方 val1449：最终留出测试。
    # 不搬动原图，不新建 split 文件。固定种子只固定划分，不固定验证抽样。
    train_all = VocDataset(root, "train")
    train_dataset, val_dataset = random_split(
        train_all, [1264, 200], generator=torch.Generator().manual_seed(0)
    )
    test_dataset = VocDataset(root, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = getVOC_dataloader(
        root="/Users/alen/Code/Pytorcc/projects/Mask-R-cnn/data",
        batch_size=2,
    )

    imageList, targets = next(iter(train_loader))

    print("batch 图片：", imageList.tensors.shape)
    print("每张图有效大小：", imageList.image_sizes)

    for target in targets:
        print("boxes：", target["boxes"].shape)
        print("labels：", target["labels"].shape)
        print("masks：", target["masks"].shape)
