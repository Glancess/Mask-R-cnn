import os

import torch

from Model.mask import Mask
from utils.loss import MaskLoss
from Dataset.split import split
from engine.train1 import train_one_epoch
from engine.val import val
from engine.test import test_best_model


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    best_path = os.path.join(root, "best.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = split(
        os.path.join(root, "data", "PennFudanPed"), batch_size=2
    )
    model = Mask().to(device)
    criterion = MaskLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_loss = float("inf")
    best_epoch = 0
    num_epochs = 40
    print(f"设备：{device} | train=120, val=25, test=25", flush=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}", flush=True)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = val(model, val_loader, criterion, device)

        # 每轮验证，loss 更低就保存最佳参数。
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)
            print(f"保存 best.pth：val loss={best_loss:.4f}", flush=True)
        else:
            print(
                f"本轮未改善，最佳为 epoch {best_epoch}，val loss={best_loss:.4f}",
                flush=True,
            )

    # 全部 epoch 结束后，加载最佳参数，只做一次 test。
    if not os.path.isfile(best_path):
        raise FileNotFoundError("找不到 best.pth，训练没有保存最佳参数")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    print(
        f"\n训练结束，加载 epoch {best_epoch} 的 best，val loss={best_loss:.4f}",
        flush=True,
    )
    test_best_model(model, test_loader, device, os.path.join(root, "testimage"))


if __name__ == "__main__":
    main()
