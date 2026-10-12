import math
import os

import torch


def save_checkpoint(save_path, model, optimizer, epoch, best_val_loss, config, split):
    """
    save_path: str；model: MaskRCNN；optimizer: SGD；epoch: int（从0开始）。
    best_val_loss: float；config / split: 普通 dict。
    无返回值，保存模型、优化器、轮次、最佳验证loss、配置和三份数据划分。
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": float(best_val_loss),
        "config": dict(config),
        "split": split,
    }
    # 先写临时文件，再替换；中途写失败时不直接破坏已有的完整 checkpoint。
    torch.save(checkpoint, save_path + ".tmp")
    os.replace(save_path + ".tmp", save_path)


def load_checkpoint(load_path, model, optimizer=None):
    """
    输入权重路径 str、MaskRCNN、可选 SGD。只加载本项目自己生成的可信文件。
    返回 checkpoint: dict；不传 optimizer 时只恢复模型，供最终 test 使用。
    """
    checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)
    required = ["model", "optimizer", "epoch", "best_val_loss", "config", "split"]
    if not isinstance(checkpoint, dict) or any(key not in checkpoint for key in required):
        raise ValueError("不是当前项目的完整 checkpoint，不能确认训练划分，请勿直接当作 test 权重。")

    # 这些改变会让训练/验证loss或网络含义发生变化，不能静默混用。
    saved_config = checkpoint["config"]
    expected = {
        "num_classes": model.num_classes,
        "min_size": model.min_size,
        "max_size": model.max_size,
        "num_rois": model.num_rois,
    }
    for key, value in expected.items():
        if saved_config.get(key) != value:
            raise ValueError(f"配置 {key} 不一致：当前={value}，checkpoint={saved_config.get(key)}。")
    if not math.isfinite(checkpoint["best_val_loss"]):
        raise ValueError("checkpoint 的 best_val_loss 不是有限数值。")

    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
