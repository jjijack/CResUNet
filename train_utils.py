import os
import shutil
import torch


def create_monthly_split(run_dates, train_days=20, val_days=5, test_days=5):
    """
    按自然月切分：每月前段训练，后段验证 + 测试。
    """
    train_idx = []
    val_idx = []
    test_idx = []

    month_map = {}
    for idx, d in enumerate(run_dates):
        if d is None:
            raise ValueError("无法从 valid_time 解析日期，请检查时间单位或时间字段。")
        key = (d.year, d.month)
        month_map.setdefault(key, []).append((d, idx))

    for key in sorted(month_map.keys()):
        items = sorted(month_map[key], key=lambda x: x[0])
        indices = [idx for _, idx in items]
        n = len(indices)
        if n == 0:
            continue

        test_count = min(test_days, n)
        val_count = min(val_days, n - test_count)
        train_count = n - val_count - test_count

        train_idx.extend(indices[:train_count])
        val_idx.extend(indices[train_count:train_count + val_count])
        test_idx.extend(indices[train_count + val_count:])

    return train_idx, val_idx, test_idx


def clear_output_dir(save_dir='./results'):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)


def total_variation_loss(img, weight=1.0):
    """平滑性约束"""
    b, t, h, w = img.size()
    tv_h = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    tv_w = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    return weight * (tv_h + tv_w) / (b * t * h * w)


def weighted_masked_rmse_loss(
    pred,
    target,
    mask,
    start_w=1.0,
    end_w=5.0,
    ignore_steps=0,
    ignore_weight=0.0,
    epsilon=1e-6,
):
    """RMSE + 时间加权"""
    diff = (pred - target) ** 2
    steps = pred.shape[1]
    weights = torch.linspace(start_w, end_w, steps, device=pred.device).view(1, -1, 1, 1)

    if ignore_steps and ignore_steps > 0:
        ignore_steps = min(ignore_steps, steps)
        weights[:, :ignore_steps] = ignore_weight

    weighted_diff = diff * weights * mask

    mse = weighted_diff.sum() / (mask.sum() + 1e-6)
    return torch.sqrt(mse + epsilon)


def smart_background_l1_loss(pred, target, mask, zero_threshold=0.1):
    """
    智能 L1：只惩罚 Ground Truth 本来就是 0 (背景) 的区域
    防止 L1 过大导致模型不敢修正真实偏差
    """
    is_background = (torch.abs(target) < zero_threshold).float()
    bg_mask = mask * is_background
    loss = (torch.abs(pred) * bg_mask).sum() / (bg_mask.sum() + 1e-6)
    return loss
