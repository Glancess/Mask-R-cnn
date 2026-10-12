import math

import torch


def encode_boxes(proposals, gt_boxes):
    """
    输入 proposals / gt_boxes: float32 Tensor[N,4]，逐行对应，都是 xyxy。
    返回 deltas: float32 Tensor[N,4]，顺序是 dx,dy,dw,dh，不是 xyxy。
    本学习版不额外乘 10/10/5/5 等权重；decode 使用同一套公式。
    注意：这两个函数只留作无权重公式练习，当前训练主流程不调用它们。
    RoISampler 使用带权重的 BoxCoder；预测必须用同一个黑盒解码，不能混用。
    """
    # 1. 左上右下坐标 -> 宽高、中心。clamp 防止除零和 log(0)。
    widths = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    heights = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
    center_x = (proposals[:, 0] + proposals[:, 2]) / 2
    center_y = (proposals[:, 1] + proposals[:, 3]) / 2

    gt_widths = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-6)
    gt_heights = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-6)
    gt_center_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_center_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2

    # 2. 位移除以 proposal 的宽高；尺寸变化取 log。
    dx = (gt_center_x - center_x) / widths
    dy = (gt_center_y - center_y) / heights
    dw = torch.log(gt_widths / widths)
    dh = torch.log(gt_heights / heights)

    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(proposals, deltas):
    """
    输入 proposals: float32 Tensor[N,4]，xyxy。
    输入 deltas: float32 Tensor[N,4]，模型预测的 dx,dy,dw,dh。
    返回 boxes: float32 Tensor[N,4]，修正后的 xyxy。
    """
    # 1. 先找原 proposal 的中心与宽高。
    widths = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    heights = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
    center_x = (proposals[:, 0] + proposals[:, 2]) / 2
    center_y = (proposals[:, 1] + proposals[:, 3]) / 2

    # 2. encode 的逆运算。限制 dw/dh 上界，避免随机初始预测使 exp 溢出。
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2].clamp(max=math.log(1000.0 / 16))
    dh = deltas[:, 3].clamp(max=math.log(1000.0 / 16))

    new_center_x = dx * widths + center_x
    new_center_y = dy * heights + center_y
    new_widths = torch.exp(dw) * widths
    new_heights = torch.exp(dh) * heights

    # 3. 注意是中心 +/- 半个宽高，不是把中心直接当左上角。
    xmin = new_center_x - new_widths / 2
    ymin = new_center_y - new_heights / 2
    xmax = new_center_x + new_widths / 2
    ymax = new_center_y + new_heights / 2

    return torch.stack([xmin, ymin, xmax, ymax], dim=1)
