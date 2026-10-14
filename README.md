# Mask R-CNN + FPN：按模块理解训练与实例分割

## 当前 VOC2012 版（2026-10-14）

- 运行 `python main.py`，仍然是每轮 train → val → 保存最佳参数，最后加载 best → test 画图。
- `Dataset/Vocdataset.py` 读取类别图、实例图，返回 boxes、labels、masks、mask_valid；模型输出21类（包含背景）。
- `Dataset/dataloader.py` 直接构造三个 loader，不调用旧的 `Dataset/split.py`，也不删除旧文件。
- 自定义学习划分：官方 train1464 → train1264/val200；官方 val1449 → 最终 held-out test。固定 seed=0，不移动原数据。
- 官方 VOC2012 test 没有公开分割 GT，因此这里的 test 不是官方测试集；不得用它挑参数或选择 best。
- 如果旧权重见过新的 val/test，不能当成独立验证、测试；应按新划分重新训练。
- Mask loss 用同一前景 proposal 分别裁剪 GT mask 和 mask_valid 到28×28，只对有效位置计算 BCE。
- 有效图的 RoIAlign 会插值：保守忽略涉及无效采样的位置；无前景或无有效像素时返回可反传的零 loss。
- `best_voc.pth` 只保存模型参数，避免覆盖以前的 `best.pth`。重新训练仍会覆盖同名 best_voc.pth，请另存需要保留的旧文件。
- 最后预测全部留出 test，前10张类别、框、实例mask对比图保存到项目根目录 `testimage/`。
- 未添加 AMP、DDP、scheduler 或 mAP。以下 PennFudan/旧模块说明保留作历史学习笔记，以本节为当前运行接口。

## 当前手写简化版运行方式（2026-10-13）

你重新手写后的文件为 `Model/backbone.py`、`FPN.py`、`anchor.py`、`mask.py`、
`utils/loss.py`、`engine/train1.py`、`engine/val.py`、`engine/test.py`。
下方旧版模块介绍保留作学习笔记，文件名、config、resume 和预训练等说明不适用于当前简化版。

- `Dataset/split.py` 固定 seed=0，返回 train120/val25/test25 三个 loader，不移动数据。
- `python main.py`：直接训练，每轮结束验证五项总 loss，变好就保存 best；全部 epoch 结束后加载 best 做一次 test。
- 验证使用 eval 和 no_grad，不更新参数和 BN 统计；不额外固定 RoI/anchor 抽样，不做 NMS 后的检测评测。
- 项目根目录的 `best.pth` 只保存验证 loss 最低的模型参数，不保存 optimizer、划分等状态。
- 每轮打印 train/val 总 loss、RPN 分类/回归、RoI 分类/回归、Mask loss，以及是否更新 best。
- 训练结束读取 best，在全部 test25 上纯预测，打印每张图的 GT/预测数量和前五个分数。
- 固定选 test loader 前 10 张，保存到项目根目录 `testimage/*_compare.png`（不按效果挑图）。
- main 没有训练/测试开关和命令行模式。
- 图左为 GT，右为预测，均画框和实例 mask；显示分数阈值默认 0.5，不计算 mAP。
- 当前没有 resume。重新训练会覆盖根目录的 best.pth，想保留旧参数请先另存。
- 划分依赖当前排序后的图片列表，训练后不要增删或重命名图片。旧 checkpoints 目录不再使用，也不自动删除。
- 当前 Backbone 随机初始化；图片不缩放，只做归一化及右下补零。不要把跑通流程等同于检测效果好。
- 这是图片级学习划分，不保证场景或人物独立。若旧模型见过这些 val/test，必须按新划分重新训练。

这个项目保留自己写的 ResNet50 Backbone、FPN、RPN 封装、RoIHeads 和分步 `train_one_epoch`。
复杂的 anchor / GT 匹配、RoI / GT 匹配、前景背景采样与 NMS 使用 torchvision 黑盒。
没有加入 AMP、DDP、scheduler、WandB，也暂时不计算 AP/mAP。

## 1. 先从哪里看

推荐阅读顺序：`Dataset/dataset.py` → `Model/BackBone.py` → `Model/BackboneWithFPN.py` → `Model/RPN.py` → `utils/RoISampler.py` → `Model/RoIHeads.py` → `utils/loss.py` → `engine.py` → `main.py`。

```text
Mask-R-cnn/
├── Model/
│   ├── BackBone.py          # ResNet50，返回C2~C5
│   ├── BackboneWithFPN.py   # C2~C5 -> P2~P6
│   ├── RPN.py               # anchor预测、RPN loss、第一阶段proposal
│   ├── RoIHeads.py          # box/classification分支与mask分支
│   └── Mask.py              # 把以上模块连接起来，提供完整预测接口
├── Dataset/
│   ├── dataset.py           # 读取原图、TXT框、每个实例的二值mask
│   └── dataloader.py        # collate_fn，返回两个list
├── utils/
│   ├── RoISampler.py        # 你封装的第二阶段匹配/采样黑盒
│   ├── loss.py              # RoI分类、框回归、mask loss
│   ├── transform.py         # 同步缩放、normalize、padding
│   ├── postprocess.py       # 修框/NMS、mask贴回原图
│   ├── split.py             # 固定train/val/test划分与检查
│   ├── checkpoint.py        # 保存/加载训练状态
│   └── boxes.py             # 无权重encode/decode公式练习，主流程不调用
├── engine.py                # 保留你的训练步骤，另加验证loss
├── main.py                  # 普通config字典与总调度
├── predict.py               # GT与预测并排，画框+半透明实例mask
├── smoke_test.py            # 两张图、一个batch的CPU流程检查
├── requirements.txt
├── data/PennFudanPed/        # 不进Git
├── checkpoints/             # 运行训练后生成，不进Git
└── outputs/test/            # 最终预测对比图，不进Git
```

## 2. 数据集与三份划分

当前本机数据检查结果：**170张图片，423个行人实例**。
全部图片的框、类别、mask数量与实例编号对应；没有发现文件内容完全相同的图片。
这里的实例数来自实际文件，不是直接照抄旧网页里的345。
数据包的 `readme.txt` 说明，发布版补标了原实验中部分小目标和严重遮挡的行人。

```text
data/PennFudanPed/
├── PNGImages/FudanPed00001.png
├── Annotation/FudanPed00001.txt
└── PedMasks/FudanPed00001_mask.png
```

原始mask里：0是背景，1、2、3……是**实例编号**，不是不同类别。
Dataset把它拆成 `[N,H,W]` 的0/1 mask，每一层是一个行人。
类别标签则都是 `person=1`，背景类别0由匹配/采样阶段产生。
TXT的左上角坐标从1开始，读取时减1；右下角作为边界保留。
框、类别和mask按实例编号排序，必须逐行对应，不能各排各的顺序。

本项目选择：

| 集合 | 图片数 | 作用 |
| --- | ---: | --- |
| train | 120 | 算loss、反向传播、SGD更新参数 |
| val | 25 | 每轮算验证loss，只用它选择best.pth |
| test | 25 | 全部训练结束后，用best权重看最终框与实例mask |

`seed=0` 固定划分，三份互不重叠、合起来覆盖所有图片。不移动原文件，只使用 `Subset`。
完整文件名和三份下标会保存进checkpoint；续训、测试使用保存的划分，不重新抽。
如果数据文件名发生变化，程序会报错，避免旧下标误指向新图片。

这是为了学习而选择的三份划分，不是官方指定的评测协议。
[PyTorch官方教程](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)示例采用120/50两份划分；这里再把留出的50张分成val25/test25。
目前只是图片级随机划分，没有人物身份/场景分组信息，不能保证人物、相邻场景之间严格独立。
如果旧权重训练时见过当前test图片，不能把它在这25张上的结果称为独立测试效果；应按新划分重新训练。

## 3. 一批训练从头到尾发生什么

```text
磁盘RGB图 + TXT框 + 实例mask
  -> Dataset: image[3,H,W] + target字典
  -> DataLoader: images列表 + targets列表
  -> 同步缩放image/box/mask -> normalize -> padding成[B,3,Hpad,Wpad]
  -> ResNet50: C2~C5 -> FPN: P2~P6
  -> RPN: anchors -> objectness/delta -> 初次修框、筛分、NMS -> proposals
  -> RoISampler: 补入GT框 -> IoU匹配 -> 前景/背景采样 -> labels与回归目标
  -> box RoIAlign 7×7 -> FC -> class_logits与box_regression
  -> 仅取正样本 -> mask RoIAlign 14×14 -> FCN -> mask_logits 28×28
  -> 五项loss相加 -> zero_grad -> backward -> SGD.step
  -> 每轮结束val loss -> 保存best/last
  -> 最后加载best -> test纯预测 -> 画框与实例mask
```

每批默认2张图片。共享特征图只提取一次，RoIAlign会按列表位置找到框所属图片的特征。
不是把每个proposal重新裁成RGB图片，再分别跑ResNet。

### 常见变量是什么类型

下面B是图片数，Ni是第i张图的proposal数，Ri是采样RoI数，Pi是正RoI数，C=2包含背景。

| 变量/函数输出 | 类型和shape |
| --- | --- |
| `dataset[i]` | `(image, target)`；image是float32 `[3,H,W]`，RGB `[0,1]` |
| `target` | boxes float32 `[N,4]` xyxy；labels int64 `[N]`；masks uint8 `[N,H,W]` |
| `collate_fn` | 两个长度B的list；图片尺寸和实例数可不同 |
| `pyramid` | dict；P2~P6都是 `[B,256,Hlevel,Wlevel]`，stride为4/8/16/32/64 |
| `rpn(...)` | `(proposals, losses)`；proposals是list，每项 `[Ni,4]` |
| `roi_sampler(...)` | 四个list：采样框 `[Ri,4]`、GT下标 `[Ri]`、类别 `[Ri]`、目标delta `[Ri,4]` |
| `forward_box(...)` | logits `[sum(Ri),C]`、delta `[sum(Ri),C*4]` |
| `get_positive_samples(...)` | 正样本框、类别、GT实例下标三个list；空的位置也保留 |
| `forward_mask(...)` | logits `[sum(Pi),C,28,28]`，尚未sigmoid |
| `train_one_epoch(...)` | float，按图片数加权的batch平均总loss |
| `evaluate_one_epoch(...)` | float，平均验证总loss |
| `model(images, compute_loss=False)` | list[dict]，每图boxes `[D,4]`、scores/labels `[D]`、masks `[D,1,H,W]` |

RPN的anchors与RoIHeads的proposals都使用图片坐标系，不要自己再除stride。
不同尺度映射由对应的黑盒处理。

## 4. 五项loss分别监督什么

总loss直接相加，当前没有额外的loss权重：

| loss | 比较双方 | 哪些样本参与 |
| --- | --- | --- |
| `loss_objectness` | RPN的物体logit vs anchor前景/背景标签；BCE | 采样的正负anchor |
| `loss_rpn_box_reg` | RPN预测delta vs anchor到匹配GT的目标delta；SmoothL1 | 正anchor |
| `loss_classifier` | RoI类别logits vs GT匹配出的类别；CrossEntropy | 采样的前景和背景RoI |
| `loss_box_reg` | RoI预测delta vs proposal到匹配GT的目标delta；SmoothL1 | 正RoI，且只取真实类别通道 |
| `loss_mask` | 真实类别通道的mask logits vs 裁剪后的GT实例mask；BCEWithLogits | 仅正RoI |

RPN采样上限每图256，最多一半正样本。第二阶段保留你的每图512上限，正样本最多25%。
上限不代表必须凑满，正样本不足时不会凭空造出正样本。
背景虽然也有匹配GT下标，但不会参与框回归和mask监督。

mask监督最重要的对应关系是：**正proposal → matched_idx → 本图某一个GT实例mask**。
再用这个proposal从该GT mask里裁出28×28目标，而不是拿整个 `[H,W]` mask直接和局部输出比较。
`BCEWithLogits`内部做sigmoid；`CrossEntropy`内部处理softmax，算loss前不要再手动做一次。

第二阶段的黑盒BoxCoder把目标编码为 `[10*dx,10*dy,5*dw,5*dh]`。
预测也必须用同一份BoxCoder解码，所以 `postprocess.py`直接复用采样器内部黑盒。
`utils/boxes.py`保留的是不带权重的学习公式，不能直接拿来解码当前模型的输出。

RPN生成proposal时，torchvision会断开框坐标那条梯度路径；NMS也不是可微操作。
因此RoI loss不会穿过proposal坐标回传到RPN预测头。
RPN头由自己的两项loss更新；RoI的三项loss仍会通过共享特征图更新Backbone/FPN。
这是当前采用的联合训练流程，不是Faster R-CNN原论文的四步交替训练。

## 5. 预处理与验证为何这样写

train / val / test共用同一份预处理，暂不做随机翻转等增强，先把基本接口学明白。
短边期望400、长边最多640，保持长宽比；尺寸取整后用实际比例同步缩放boxes。
RGB图双线性缩放，实例mask最近邻缩放，避免凭空产生新实例编号。
然后使用ImageNet的RGB mean `[0.485,0.456,0.406]`、std `[0.229,0.224,0.225]`归一化。
最后右/下补零到64的倍数，匹配P6的stride；真实尺寸另存，RPN不能把补零区当成真实图片。

默认batch=2，`train_one_epoch`固定ResNet BN的running mean/var，避免小batch统计剧烈波动。
这不是冻结整个Backbone，卷积和BN的weight/bias仍然参与学习。

验证仍然要GT匹配与采样，才能计算和训练相同的五项loss。
它是 `eval + no_grad`，没有 `backward/step`，不更新参数或BN统计。
RPN/第二阶段随机采样固定种子，并在验证后恢复训练随机状态。
所以：**验证loss不是先NMS后与GT框算一个loss；那是另一条检测评价流程。**

## 6. 最后怎么得到框与实例分割

纯预测不传targets：`model.eval()`，并在 `torch.no_grad()` 中调用模型。

```text
RPN proposals
  -> box branch预测类别和delta
  -> 同一BoxCoder修框 -> 分数过滤 -> 按类别NMS -> 最终检测框
  -> 用最终框跑mask branch
  -> 每个框选择预测类别的mask通道 -> sigmoid
  -> 28×28 mask缩放/贴回原图，框恢复原图坐标
  -> 与GT并排画框和半透明mask
```

训练mask分支用的是**正样本proposal**；预测mask分支用的是**NMS后的最终框**，不要混淆。
输出mask是 `[D,1,H,W]` 的概率图，画图时以0.5二值化；每个实例不同颜色。
GT只用于对照图左边，不参与模型预测。没有检测结果时，右边保留原图，不会报错。

## 7. 运行方式与checkpoint

在项目目录打开终端，使用已经安装匹配版本torch/torchvision的Python环境。
本机实际检查环境为torch 2.12.1 / torchvision 0.27.1，CPU；没有验证CUDA显存或完整训练精度。
如需安装基础依赖，可以使用 `python -m pip install -r requirements.txt`；服务器应使用彼此匹配的CUDA版本。

先做轻量检查（只用两张图、一个训练batch；不下载预训练权重）：

```bash
python -B smoke_test.py
```

正式训练前查看 `main.py` 的config，确认数据路径、输出目录、batch、尺寸、轮数和学习率。
默认只给ResNet50加载ImageNet预训练权重，首次正式训练可能下载它；FPN/RPN/RoIHeads都是新初始化。
这不是加载完整COCO Mask R-CNN后直接微调，不能期待少量训练就得到官方教程那样的效果。

```bash
python -B main.py train    # 从头训练，每轮val，最后加载best画test
python -B main.py resume   # 读取last继续；num_epochs是目标总轮数
python -B main.py test     # 已有best时直接画test，不再训练
```

生成的图片在 `outputs/test/*_compare.png`：左GT、右预测，两边都画框和实例mask。
`num_test_images=None`画全部25张，改为5只看前5张。

`best.pth`保存val总loss最低时的模型；`last.pth`保存最后完成的epoch，通常用于续训。
两者都包含model、optimizer、epoch、best_val_loss、config以及三份数据划分。
加载时检查模型关键配置和数据文件名；不要拿缺少划分的旧裸state_dict冒充独立test权重。
从头训练遇到已存在的best/last会报错，避免静默覆盖。新实验应换save_dir，建议同时换output_dir。
resume会恢复优化器里的lr和momentum，不会自动按新config重设；当前不保存RNG和DataLoader的精确进度，因此不是逐位一致的断点重放。

## 8. 当前检查范围与限制

已做本地数据数量/shape/label/框范围/mask实例对应检查，轻量真实反传、参数更新、可重复验证、checkpoint往返、原图mask还原与空结果检查。
smoke只证明接口与控制流可执行，不证明收敛、检测精度或论文复现指标。
smoke的随机初始化预测图只用于检查流程，不能作为训练效果图展示。
正式main保持原来的RPN候选上限：训练2000、预测1000；smoke临时减少候选框和图片尺寸以加快CPU检查。

目前只有person一个前景类别、小规模图片级划分、验证总loss和最终可视化。
loss最低不一定对应检测/分割精度最高；AP/mAP、增强、学习率调度和更严格的分组评测可以以后学习时再加，不属于这次必须修复的接口问题。
不要看test结果来反复选择checkpoint、调阈值并宣称无偏测试；需要调参时应使用val。

`.gitignore`排除原始数据、压缩包、checkpoint、预测结果、缓存、`__pycache__`、虚拟环境与环境变量文件。
本次修改前的源码备份为 `/private/tmp/maskrcnn-current-MJKoXn/source_before.tar.gz`，不包含数据集和权重；临时目录不是永久备份位置。
本次未进行git add、commit或push，原有未提交改动仍保留。
