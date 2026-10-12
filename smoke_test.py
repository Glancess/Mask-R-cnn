import math
import os
import tempfile

import torch
from torch.utils.data import DataLoader, Subset

from Dataset.dataset import PennFudanDataset
from Dataset.dataloader import collate_fn
from Model.Mask import MaskRCNN
from engine import train_one_epoch, evaluate_one_epoch
from predict import save_comparison
from utils.loss import box_loss, mask_loss
from utils.transform import prepare_images
from utils.split import make_split, check_split
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.postprocess import postprocess_boxes, restore_predictions


def main():
    """只做CPU两张图、一个batch检查；不下载权重，不开始完整训练，无返回值。"""
    torch.set_num_threads(2)
    torch.manual_seed(0)
    device = torch.device("cpu")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = PennFudanDataset(os.path.join(project_dir, "data", "PennFudanPed"))
    split = make_split(dataset)
    check_split(dataset, split)
    assert split == make_split(dataset)
    print("划分:", {key: len(split[key]) for key in ["train", "val", "test"]}, flush=True)

    loader = DataLoader(Subset(dataset, [0, 1]), batch_size=2, collate_fn=collate_fn)
    images, targets = next(iter(loader))
    old_boxes = [target["boxes"].clone() for target in targets]
    batch, resized_targets, shapes, _ = prepare_images(images, targets, 128, 192)
    assert batch.shape[-2] % 64 == 0 and batch.shape[-1] % 64 == 0
    assert all(torch.equal(target["boxes"], old) for target, old in zip(targets, old_boxes))
    for target, shape in zip(resized_targets, shapes):
        assert target["masks"].shape[-2:] == shape
        assert set(target["masks"].unique().tolist()) <= {0, 1}
    print("batch:", tuple(batch.shape), "真实尺寸:", shapes, flush=True)

    model = MaskRCNN(
        weights=None, min_size=128, max_size=192, num_rois=8,
        score_thresh=0.0, detections_per_img=4,
    )
    # 只在smoke里缩小RPN候选数，正式main不改你的2000/1000配置。
    model.rpn.rpn._pre_nms_top_n = {"training": 64, "testing": 32}
    model.rpn.rpn._post_nms_top_n = {"training": 16, "testing": 8}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

    # 1. 同一个BoxCoder编码和解码，验证10/10/5/5权重没有混用。
    proposals = [torch.tensor([[10., 20., 50., 70.]])]
    gt = [torch.tensor([[12., 18., 55., 72.]])]
    coder = model.roi_sampler.roi_heads.box_coder
    encoded = coder.encode(gt, proposals)
    decoded = coder.decode(torch.cat(encoded), proposals)[:, 0]
    assert torch.allclose(decoded, gt[0], atol=1e-5)

    # 两个同位置的高分person框，NMS后只留一个，背景通道不输出。
    repeated_boxes = [torch.tensor([[10., 20., 50., 70.], [10., 20., 50., 70.]])]
    results = postprocess_boxes(
        torch.tensor([[-10., 10.], [-10., 9.]]), torch.zeros((2, 8)),
        repeated_boxes, [(100, 100)], model.roi_sampler,
    )
    assert results[0]["boxes"].shape == (1, 4) and results[0]["labels"].tolist() == [1]

    # 人工构造mask logits，仅检查类别通道、贴回位置和空图片不串位，不代表模型效果。
    logits = torch.full((1, 2, 28, 28), -20.)
    logits[:, 1] = 20.
    empty_result = {"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.int64),
                    "scores": torch.empty(0)}
    restored = restore_predictions(
        [empty_result, results[0]], logits, [(100, 100), (100, 100)], [(200, 200), (200, 200)],
    )
    assert restored[0]["masks"].shape == (0, 1, 200, 200)
    assert torch.allclose(restored[1]["boxes"], repeated_boxes[0][:1] * 2)
    assert restored[1]["masks"][0, 0, 80, 60] > 0.99
    assert restored[1]["masks"][0, 0, 0, 0] == 0

    # 2. 真正调用你的train_one_epoch，检查反传和SGD是否更新了卷积参数。
    before = model.backbone.backbone.conv1.weight.detach().clone()
    train_loss = train_one_epoch(
        model.backbone, model.rpn, model.roi_sampler, model.roi_heads,
        loader, optimizer, device, box_loss, mask_loss, min_size=128, max_size=192,
    )
    assert math.isfinite(train_loss)
    assert not torch.equal(before, model.backbone.backbone.conv1.weight)
    for module in [model.backbone, model.rpn, model.roi_heads]:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g).item() for g in grads)
    assert model.roi_heads.box_roi_pool.scales == [0.25, 0.125, 0.0625, 0.03125]

    # 3. 验证不更新参数/BN统计，不扰动训练RNG，两次验证loss应完全一致。
    versions = [p._version for p in model.parameters()]
    buffers = {key: value.clone() for key, value in model.named_buffers()}
    rng = torch.random.get_rng_state().clone()
    val_losses = []
    for _ in range(2):
        val_losses.append(evaluate_one_epoch(
            model.backbone, model.rpn, model.roi_sampler, model.roi_heads,
            loader, device, box_loss, mask_loss, min_size=128, max_size=192,
        ))
    assert val_losses[0] == val_losses[1] and math.isfinite(val_losses[0])
    assert versions == [p._version for p in model.parameters()]
    assert all(torch.equal(value, buffers[key]) for key, value in model.named_buffers())
    assert torch.equal(rng, torch.random.get_rng_state())
    print("train_loss:", train_loss, "val_loss（可重复）:", val_losses[0], flush=True)

    # 4. 权重写到临时目录，再打乱一层参数并加载回来，避免只测save不测load。
    output_dir = tempfile.mkdtemp(prefix="maskrcnn-smoke-")
    checkpoint_path = os.path.join(output_dir, "best.pth")
    config = {"num_classes": 2, "min_size": 128, "max_size": 192, "num_rois": 8}
    save_checkpoint(checkpoint_path, model, optimizer, 0, val_losses[0], config, split)
    saved_weight = model.roi_heads.cls_score.weight.detach().clone()
    with torch.no_grad():
        model.roi_heads.cls_score.weight.add_(1)
    checkpoint = load_checkpoint(checkpoint_path, model, optimizer)
    assert torch.equal(model.roi_heads.cls_score.weight, saved_weight)
    assert checkpoint["split"] == split and checkpoint["epoch"] == 0
    del checkpoint

    # 5. 纯预测：不传GT；检查框和mask确实回到原图尺寸。
    model.eval()
    with torch.no_grad():
        predictions = model(images, compute_loss=False)
    for prediction, image in zip(predictions, images):
        n = len(prediction["boxes"])
        h, w = image.shape[-2:]
        assert prediction["boxes"].shape == (n, 4)
        assert prediction["masks"].shape == (n, 1, h, w)
        assert (prediction["labels"] == 1).all()
        assert torch.isfinite(prediction["masks"]).all()
        assert (prediction["boxes"] >= 0).all()
        assert (prediction["boxes"][:, 0::2] <= w + 1e-4).all()
        assert (prediction["boxes"][:, 1::2] <= h + 1e-4).all()
    print("预测shape:", [{k: tuple(v.shape) for k, v in p.items()} for p in predictions], flush=True)
    save_path = save_comparison(images[0], targets[0], predictions[0], os.path.join(output_dir, "compare.png"))

    # 6. 没有检测框/没有GT时也要能跑，而不是在cat或mask还原处报错。
    model.roi_sampler.roi_heads.score_thresh = 1.0
    empty_target = {
        "boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64),
        "masks": torch.empty((0, *images[0].shape[-2:]), dtype=torch.uint8),
    }
    with torch.no_grad():
        empty = model([images[0]], compute_loss=False)[0]
        empty_gt_losses = model([images[0]], [empty_target], compute_loss=True)
        mixed_losses = model(images, [empty_target, targets[1]], compute_loss=True)
    assert empty["boxes"].shape == (0, 4)
    assert empty["masks"].shape == (0, 1, *images[0].shape[-2:])
    assert all(torch.isfinite(value).item() for value in empty_gt_losses.values())
    assert all(torch.isfinite(value).item() for value in mixed_losses.values())
    print("Smoke test PASSED：train/eval、梯度、checkpoint、预测和空结果。", flush=True)
    print("随机初始化的流程检查图，不代表训练效果：", save_path, flush=True)


if __name__ == "__main__":
    main()
