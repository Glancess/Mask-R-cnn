import os

import torch

from Model.mask import Mask
from utils.loss import MaskLoss
from Dataset.dataloader import getVOC_dataloader
from engine.train1 import train_one_epoch
from engine.val import val
from engine.test import test_best_model


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    best_path = os.path.join(root, "best_voc.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = getVOC_dataloader(
        os.path.join(root, "data"), batch_size=2, flip_prob=0.5
    )
    print("使用自己的Mask模型：ResNet50 + FPN，VOC共21类（含背景）。", flush=True)
    model = Mask().to(device)
    criterion = MaskLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0005,
        momentum=0.9,
        weight_decay=0.0005,  # 小数据集加一点正则，尝试减轻过拟合。
    )
    best_loss = float("inf")
    best_epoch = 0
    num_epochs = 100
    print(
        f"设备：{device} | train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, held-out test={len(test_loader.dataset)}",
        flush=True,
    )

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}", flush=True)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = val(model, val_loader, criterion, device)

        # 每轮验证，loss 更低就保存最佳参数。
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)
            print(f"保存 best_voc.pth：val loss={best_loss:.4f}", flush=True)
        else:
            print(
                f"本轮未改善，最佳为 epoch {best_epoch}，val loss={best_loss:.4f}",
                flush=True,
            )

    # 全部 epoch 结束后，加载最佳参数，只做一次 test。
    if not os.path.isfile(best_path):
        raise FileNotFoundError("找不到 best_voc.pth，训练没有保存最佳参数")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    print(
        f"\n训练结束，加载 epoch {best_epoch} 的 best，val loss={best_loss:.4f}",
        flush=True,
    )
    test_best_model(model, test_loader, device, os.path.join(root, "testimage"))


if __name__ == "__main__":
    main()
