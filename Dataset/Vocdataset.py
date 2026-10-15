import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import functional as F


class VocDataset(Dataset):
    def __init__(self, root, image_set="train"):
        self.root = root

        # 官方接口负责读取图片列表、RGB图片、类别分割图。
        self.voc = VOCSegmentation(
            root=root,
            year="2012",
            image_set=image_set,
            download=False,
        )

        # 保留文件名，方便后面保存预测对比图。
        self.imgs = [os.path.basename(path) for path in self.voc.images]

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        image, class_mask = self.voc[idx]

        # 类别图：像素值表示类别，不转成 RGB。
        class_mask = np.array(class_mask)
        """
        0   0   1   1   0
        0   1   1   1   0
        0   0   0  15  15
        0   0  255 15  15
        但是并不知道insance的划分
        """

        # 实例图：像素值表示属于第几个物体。
        name = os.path.splitext(self.imgs[idx])[0]
        object_path = os.path.join(
            self.root,
            "VOCdevkit",
            "VOC2012",
            "SegmentationObject",
            name + ".png",
        )
        object_mask = np.array(Image.open(object_path))
        # object_mask.shape  # [H,W]
        # 255 表示未标注区域，后面计算 Mask loss 时需要忽略。
        mask_valid = (class_mask != 255) & (object_mask != 255)

        boxes = []
        labels = []
        masks = []

        # 遍历这张图片里的实例编号。
        for obj_id in np.unique(object_mask):
            if obj_id == 0 or obj_id == 255:
                continue

            # 得到当前物体的二值 mask。
            mask = (object_mask == obj_id) & mask_valid
            if not mask.any():
                continue

            # 到类别图里，查询这个物体的类别。
            class_ids = np.unique(class_mask[mask])
            class_ids = class_ids[(class_ids > 0) & (class_ids <= 20)]
            if len(class_ids) != 1:
                raise ValueError(f"{name}：实例类别不明确")

            # 从这个实例的 mask 计算紧框，保证三项对应同一个物体。
            y, x = np.where(mask)
            boxes.append(
                [
                    int(x.min()),
                    int(y.min()),
                    int(x.max()) + 1,
                    int(y.max()) + 1,
                ]
            )
            labels.append(int(class_ids[0]))
            masks.append(torch.as_tensor(mask, dtype=torch.uint8))

        h, w = object_mask.shape

        if len(masks) > 0:
            masks = torch.stack(masks)
        else:
            masks = torch.empty((0, h, w), dtype=torch.uint8)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": masks,
            "mask_valid": torch.as_tensor(mask_valid, dtype=torch.bool),
        }

        return F.to_tensor(image), target


if __name__ == "__main__":
    dataset = VocDataset(
        root="/Users/alen/Code/Pytorcc/projects/Mask-R-cnn/data",
        image_set="train",
    )

    image, target = dataset[0]

    print(image.shape)
    print(target)
