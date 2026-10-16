# Mask R-CNN + FPN

PyTorch实例分割学习项目。默认运行自己组织的 `Model/mask.py`：ResNet50 + FPN + RPN + 检测/Mask分支，输出VOC的20类物体和背景。Backbone加载ImageNet预训练权重，匹配、抽样、RoIAlign和NMS复用torchvision。

## 文件目录

```text
Mask-R-cnn/
├── main.py                  # 训练 → 每轮验证 → 保存best → 最后测试、画图
├── requirements.txt         # 依赖
├── README.md                # 项目说明
├── EXPERIMENT_LOG.md         # 参数调整与实验记录
├── .gitignore               # 排除数据、权重、日志、缓存等
├── Model/
│   ├── mask.py              # 当前模型，连接Backbone、FPN、RPN和两个分支
│   ├── backbone.py          # ResNet50，提取四级特征
│   ├── fpn.py               # 特征融合，生成五级特征图
│   ├── anchor.py            # RPN封装：生成proposal、计算RPN loss
│   └── mask_coco.py         # 完整COCO预训练对照版，当前main不使用
├── Dataset/
│   ├── Vocdataset.py        # VOC图片、类别/实例mask → image和GT字典
│   ├── dataloader.py        # 数据划分、归一化、补零、构造loader
│   ├── dataset.py           # 旧PennFudan数据集，当前不使用
│   └── split.py             # 旧PennFudan划分，当前不使用
├── engine/
│   ├── train1.py            # 一轮训练：算loss、反传、SGD更新
│   ├── val.py               # 一轮验证：只算loss，不更新参数
│   └── test.py              # 预测、画框与实例mask、保存GT/预测对比图
├── utils/
│   └── loss.py              # RoI分类、框回归、忽略255区域的Mask loss
└── Test.py                  # stack/cat等Tensor操作练习
```

## 数据集

使用VOC2012，下载 [VOCtrainval_11-May-2012.tar](https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 后解压到项目的 `data/`：

```text
data/VOCdevkit/VOC2012/
├── JPEGImages/              # RGB图片
├── SegmentationClass/       # 类别图：像素表示类别
├── SegmentationObject/      # 实例图：像素表示物体编号
└── ImageSets/Segmentation/
    ├── train.txt
    └── val.txt
```

Dataset返回 `image [3,H,W]` 和target字典：`boxes [N,4]`、`labels [N]`、`masks [N,H,W]`、`mask_valid [H,W]`。框由实例mask计算；255是忽略区域。DataLoader不缩放图片，只做归一化、右下补零。

## 数据划分

| 本项目集合 | 来源 | 数量 |
| --- | --- | ---: |
| train | 官方分割train的一部分 | 1264 |
| val | 官方分割train中留出 | 200 |
| test | 官方分割val | 1449 |

固定 `seed=0`，只划分索引，不移动图片。train随机水平翻转，val/test不增强。这里的test是自定义留出集，不是VOC官方test；不得用它反复调参。

## 运行

在项目根目录、已安装匹配torch/torchvision的环境中执行：

```bash
python -m pip install -r requirements.txt
python -u main.py
```

默认有CUDA用CUDA，否则用CPU。学习率、batch size和epoch直接在 `main.py` 修改。训练结束加载 `best_voc.pth`，预测全部test，前10张GT/预测对比图保存到 `testimage/`。重新训练会覆盖同名best，请先备份旧权重；数据、权重和图片不随仓库提供。

当前只按loss验证和可视化，尚未计算box/mask AP；训练与验证差距、RoI分类/回归表现仍有待改善。低Mask loss不代表最终检测和分割效果一定好。
