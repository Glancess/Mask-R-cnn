import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet50_Weights

from Dataset.dataset import PennFudanDataset
from Dataset.dataloader import collate_fn
from Model.Mask import MaskRCNN
from engine import train_one_epoch, evaluate_one_epoch
from predict import test_best_model
from utils.loss import box_loss, mask_loss
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.split import make_split, check_split

# =========================================================
# 1. 配置：先用普通字典，不引入配置框架。
# 路径以 main.py 所在目录为准，不依赖你从哪个目录运行。
# =========================================================
project_dir = os.path.dirname(os.path.abspath(__file__))
config = {
    "data_root": os.path.join(project_dir, "data", "PennFudanPed"),
    "save_dir": os.path.join(project_dir, "checkpoints"),
    "output_dir": os.path.join(project_dir, "outputs", "test"),
    "num_classes": 2,  # 背景0 + person1，不是两个前景类别
    "pretrained": True,  # 仅加载 ImageNet ResNet50，不是完整COCO检测权重
    "batch_size": 2,
    "num_workers": 0,  # 先跑通；服务器可改为2或4
    "num_epochs": 40,  # 总轮数；resume不是再额外训练20轮
    "lr": 0.001,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "min_size": 400,
    "max_size": 640,
    "num_rois": 512,  # 保留你原来的每图采样上限，最多25%前景
    "num_val": 25,
    "num_test": 25,  # 170张 -> train120 / val25 / test25
    "seed": 0,
    "score_thresh": 0.5,
    "nms_thresh": 0.5,
    "detections_per_img": 100,
    "num_test_images": None,  # None画全部test；改为5就只画前5张
}


def main(mode="train"):
    """
    mode: str，train=从头训练，resume=接last继续，test=直接加载best画图。
    无返回值。负责串联 Dataset -> 训练 -> 验证 -> checkpoint -> 最终test。
    """
    if mode not in ["train", "resume", "test"]:
        raise ValueError("mode 只能是 train / resume / test。")
    if config["num_epochs"] <= 0 or config["batch_size"] <= 0:
        raise ValueError("num_epochs 和 batch_size 必须大于0。")

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # 优先使用服务器CUDA；没CUDA时用CPU，避免假定所有设备都支持检测算子。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_path = os.path.join(config["save_dir"], "best.pth")
    last_path = os.path.join(config["save_dir"], "last.pth")

    # 从头训练不能悄悄覆盖上一次的权重；换目录或显式选择resume。
    if mode == "train" and (os.path.exists(best_path) or os.path.exists(last_path)):
        raise FileExistsError(
            "save_dir 已有权重。继续训练用 resume；新实验请修改 save_dir。"
        )
    if mode == "resume" and not (
        os.path.isfile(last_path) and os.path.isfile(best_path)
    ):
        raise FileNotFoundError("resume 需要同一次训练留下的 last.pth 和 best.pth。")
    if mode == "test" and not os.path.isfile(best_path):
        raise FileNotFoundError(
            f"找不到 {best_path}，先训练或把正确的best.pth放到save_dir。"
        )

    # =====================================================
    # 2. Dataset + 模块，MaskRCNN只是把你写的模块放在一起。
    # 训练仍然调用你原来分模块传参的 train_one_epoch。
    # =====================================================
    dataset = PennFudanDataset(config["data_root"])
    weights = (
        ResNet50_Weights.DEFAULT if mode == "train" and config["pretrained"] else None
    )
    model = MaskRCNN(
        num_classes=config["num_classes"],
        weights=weights,
        min_size=config["min_size"],
        max_size=config["max_size"],
        num_rois=config["num_rois"],
        score_thresh=config["score_thresh"],
        nms_thresh=config["nms_thresh"],
        detections_per_img=config["detections_per_img"],
    ).to(device)

    # 模块都在同一个model里，parameters()就包含Backbone/FPN/RPN/两个RoI分支。
    optimizer = (
        torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
        if mode != "test"
        else None
    )

    start_epoch = 0
    best_val_loss = float("inf")
    if mode == "train":
        split = make_split(
            dataset, config["num_val"], config["num_test"], config["seed"]
        )
    else:
        load_path = best_path if mode == "test" else last_path
        checkpoint = load_checkpoint(load_path, model, optimizer)
        split = checkpoint["split"]
        check_split(dataset, split)
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"已加载 {load_path}，保存于第 {checkpoint['epoch'] + 1} 轮。")
        del checkpoint

        if mode == "resume":
            # 在更新任何参数前检查best/last是否匹配，不能训练完才发现拿混了。
            best_checkpoint = torch.load(
                best_path, map_location="cpu", weights_only=True
            )
            if (
                best_checkpoint.get("split") != split
                or best_checkpoint.get("best_val_loss") != best_val_loss
                or best_checkpoint.get("epoch", start_epoch) >= start_epoch
            ):
                raise ValueError(
                    "best/last不是同一套训练状态，请恢复正确文件或换目录重新训练。"
                )
            for key in ["num_classes", "min_size", "max_size", "num_rois"]:
                if best_checkpoint.get("config", {}).get(key) != config[key]:
                    raise ValueError(f"best/last的配置 {key} 不一致。")
            del best_checkpoint

            # 优化器恢复的是原来正在使用的参数，而不是新config里的初始值。
            for key in ["lr", "momentum", "weight_decay"]:
                config[key] = optimizer.param_groups[0][key]
            if start_epoch >= config["num_epochs"]:
                print("已经达到目标总轮数，不再训练，直接加载best画test。")

    print(
        f"device={device} | train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}"
    )

    # 直接看已有best：不会进入训练循环，也不会用test选择checkpoint。
    if mode == "test":
        test_best_model(
            model,
            dataset,
            split["test"],
            config["output_dir"],
            device,
            num_images=config["num_test_images"],
        )
        return

    # =====================================================
    # 3. 每轮只读取 train / val；test 暂时不进DataLoader。
    # =====================================================
    train_loader = DataLoader(
        Subset(dataset, split["train"]),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, split["val"]),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )

    for epoch in range(start_epoch, config["num_epochs"]):
        train_loss = train_one_epoch(
            model.backbone,
            model.rpn,
            model.roi_sampler,
            model.roi_heads,
            train_loader,
            optimizer,
            device,
            box_loss,
            mask_loss,
            epoch=epoch,
            min_size=config["min_size"],
            max_size=config["max_size"],
        )
        val_loss = evaluate_one_epoch(
            model.backbone,
            model.rpn,
            model.roi_sampler,
            model.roi_heads,
            val_loader,
            device,
            box_loss,
            mask_loss,
            min_size=config["min_size"],
            max_size=config["max_size"],
        )
        print(
            f"Epoch {epoch + 1}/{config['num_epochs']} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        # best：截至当前val loss最低；last：最后一次完整跑完的epoch。
        # test完全不参与这里的比较。
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_path, model, optimizer, epoch, best_val_loss, config, split
            )
            print(f"保存best.pth，val_loss={best_val_loss:.4f}")
        save_checkpoint(
            last_path, model, optimizer, epoch, best_val_loss, config, split
        )

    # =====================================================
    # 4. 全部训练结束后，重新加载val上最好的模型，最后才看test。
    # =====================================================
    checkpoint = load_checkpoint(best_path, model)
    check_split(dataset, checkpoint["split"])
    if checkpoint["split"] != split:
        raise ValueError("best和last的数据划分不一致，不能作为同一次实验继续。")
    del checkpoint
    test_best_model(
        model,
        dataset,
        split["test"],
        config["output_dir"],
        device,
        num_images=config["num_test_images"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="简单的 Mask R-CNN 训练与可视化")
    parser.add_argument(
        "mode", nargs="?", default="train", choices=["train", "resume", "test"]
    )
    args = parser.parse_args()
    main(args.mode)
