import os
import re

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class PennFudanDataset(Dataset):
    def __init__(self, root):
        """root: str或Path，指向同时包含PNGImages/PedMasks/Annotation的目录。"""
        self.root = root

        self.img_dir = os.path.join(root, "PNGImages")
        self.mask_dir = os.path.join(root, "PedMasks")
        self.ann_dir = os.path.join(root, "Annotation")

        self.imgs = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".png")])

    def __len__(self):
        """返回 int：图片数量，不是行人实例数量。"""
        return len(self.imgs)

    def read_boxes_from_txt(self, txt_path):
        """
        输入 txt_path: str或Path，单张图片的Annotation文件路径。
        从 Annotation/*.txt 中读取：
        Bounding box for object 1 ... : (160, 182) - (302, 431)

        return:
            boxes: float32 Tensor[N,4]，xyxy，按实例编号升序。
            TXT左上角从1开始；Tensor图片坐标从0开始，xmin/ymin要减1。
            xmax/ymax保留，表示框的右/下边界，不是最后一个像素的下标。
        """

        # key: object index
        # value: [xmin, ymin, xmax, ymax]
        box_dict = {}

        pattern = re.compile(
            r"Bounding box for object\s+(\d+).*?"
            r"\((\d+),\s*(\d+)\)\s*-\s*"
            r"\((\d+),\s*(\d+)\)"
        )

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = pattern.search(line)

                if match is not None:
                    obj_id = int(match.group(1))

                    xmin = int(match.group(2))
                    ymin = int(match.group(3))
                    xmax = int(match.group(4))
                    ymax = int(match.group(5))

                    box_dict[obj_id] = [xmin - 1, ymin - 1, xmax, ymax]

        # 按 object index 排序
        boxes = [box_dict[obj_id] for obj_id in sorted(box_dict.keys())]

        return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    def read_instance_masks(self, mask_path):
        """
        输入 mask_path: str或Path，单张图片的实例mask PNG路径。
        原始 mask:
            0 = background
            1 = person instance 1
            2 = person instance 2
            ...

        转换成:
            [N, H, W]

        每一张都是 0/1 binary mask。
        返回 tuple(masks, obj_ids)：
            masks: uint8 Tensor[N,H,W]，每一层对应一个行人实例。
            obj_ids: numpy.ndarray[N]，原始实例编号，升序；不是类别标签。
        """

        mask = np.array(Image.open(mask_path))

        # 例如 [0, 1, 2]
        obj_ids = np.unique(mask)

        # 去掉 background = 0
        obj_ids = obj_ids[obj_ids != 0]

        # [H, W] -> [N, H, W]
        masks = mask == obj_ids[:, None, None]

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        return masks, obj_ids

    def __getitem__(self, idx):
        """
        输入 idx: int，图片下标。
        返回 tuple(image, target)：
            image: float32 Tensor[3,H,W]，RGB，范围[0,1]，尚未normalize。
            target: dict[str, Tensor]，三项逐行对应同一个实例：
                boxes:  float32 [N,4]，xyxy
                labels: int64 [N]，全部为person=1
                masks:  uint8 [N,H,W]，值为0或1
        """
        # -----------------------------
        # 1. 文件路径
        # -----------------------------

        img_name = self.imgs[idx]

        # FudanPed00001.png -> FudanPed00001
        stem = os.path.splitext(img_name)[0]

        img_path = os.path.join(self.img_dir, img_name)

        mask_path = os.path.join(self.mask_dir, f"{stem}_mask.png")

        ann_path = os.path.join(self.ann_dir, f"{stem}.txt")

        # -----------------------------
        # 2. image
        # -----------------------------

        image = Image.open(img_path).convert("RGB")

        # PIL -> Tensor
        # [H, W, 3] -> [3, H, W]
        # uint8 0~255 -> float 0~1
        image = F.to_tensor(image)

        # -----------------------------
        # 3. GT boxes
        # -----------------------------

        boxes = self.read_boxes_from_txt(ann_path)

        # -----------------------------
        # 4. GT instance masks
        # -----------------------------

        masks, obj_ids = self.read_instance_masks(mask_path)

        # -----------------------------
        # 5. labels
        # -----------------------------

        num_objs = boxes.shape[0]

        # PennFudan 只有 person
        # 0 = background
        # 1 = person
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # -----------------------------
        # 6. sanity check
        # -----------------------------

        if boxes.shape[0] != masks.shape[0]:
            raise ValueError(
                f"{img_name}: "
                f"boxes={boxes.shape[0]}, "
                f"masks={masks.shape[0]}, "
                f"mask obj_ids={obj_ids}"
            )

        # -----------------------------
        # 7. target
        # -----------------------------

        target = {
            "boxes": boxes,  # [N, 4]
            "labels": labels,  # [N]
            "masks": masks,  # [N, H, W]
        }

        return image, target


# =========================================================
# 主函数
# =========================================================

if __name__ == "__main__":

    # 用脚本所在位置找数据，从项目根目录运行也不会多拼一层projects。
    root = os.path.join(os.path.dirname(__file__), "..", "data", "PennFudanPed")

    dataset = PennFudanDataset(root)

    print("dataset size:", len(dataset))

    image, target = dataset[0]

    print("\n========== IMAGE ==========")
    print("image shape:", image.shape)
    print("image dtype:", image.dtype)
    print("image min:", image.min().item())
    print("image max:", image.max().item())

    print("\n========== BOXES ==========")
    print("boxes shape:", target["boxes"].shape)
    print(target["boxes"])

    print("\n========== LABELS ==========")
    print("labels shape:", target["labels"].shape)
    print(target["labels"])

    print("\n========== MASKS ==========")
    print("masks shape:", target["masks"].shape)
    print("masks dtype:", target["masks"].dtype)

    for i in range(target["masks"].shape[0]):
        print(
            f"instance {i}:",
            "unique =",
            torch.unique(target["masks"][i]),
            "foreground pixels =",
            target["masks"][i].sum().item(),
        )
