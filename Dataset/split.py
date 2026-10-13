import torch
from torch.utils.data import DataLoader, random_split

from .dataset import PennFudanDataset
from .dataloader import collate_fn


def split(root, batch_size=2):
    """输入数据路径；返回 train、val、test 三个 loader，不移动图片。"""
    dataset = PennFudanDataset(root)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [120, 25, 25],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
