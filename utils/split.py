import torch


def make_split(dataset, num_val=25, num_test=25, seed=0):
    """
    输入 Dataset、验证/测试图片数 int、随机种子 int。
    返回 dict：全部文件名，以及 train / val / test 三个互不重叠的下标列表。
    只记录划分，不复制、移动或修改 data 里的原文件。
    """
    if num_val <= 0 or num_test <= 0 or num_val + num_test >= len(dataset):
        raise ValueError("train / val / test 都要至少有一张图片。")

    # 用独立 Generator，不改变训练用的全局随机状态。
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    num_train = len(dataset) - num_val - num_test
    split = {
        "image_names": list(dataset.imgs),
        "train": indices[:num_train],
        "val": indices[num_train:num_train + num_val],
        "test": indices[num_train + num_val:],
        "seed": seed,
    }
    check_split(dataset, split)
    return split


def check_split(dataset, split):
    """输入 Dataset 和已保存的划分 dict；检查不通过就报错，无返回值。"""
    required = ["image_names", "train", "val", "test", "seed"]
    if not isinstance(split, dict) or any(key not in split for key in required):
        raise ValueError("checkpoint缺少完整train/val/test划分，不能当作当前独立测试权重。")
    if list(dataset.imgs) != split["image_names"]:
        raise ValueError("数据集文件名发生变化，不能直接套用旧 checkpoint 的划分。")

    indices = split["train"] + split["val"] + split["test"]
    if any(len(split[name]) == 0 for name in ["train", "val", "test"]):
        raise ValueError("train / val / test 不能有空集合。")
    if sorted(indices) != list(range(len(dataset))):
        raise ValueError("划分有重复、遗漏或越界下标，不能继续训练/测试。")
