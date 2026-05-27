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


def total_variation_loss(img, mask=None, weight=1.0):
    """平滑性约束。支持 mask 避免海岸线陆地边界产生虚假梯度惩教。"""
    b, t, h, w = img.size()
    # 水平方向：只有相邻两像素都在 mask 内才计入
    if mask is not None:
        # mask: (B, 1, H, W) — 广播到所有时间步
        if mask.dim() == 4 and mask.shape[1] == 1:
            m = mask.float()
        else:
            m = mask.float()
        m_h = m[:, :, :, 1:] * m[:, :, :, :-1]
        m_w = m[:, :, 1:, :] * m[:, :, :-1, :]
    else:
        m_h = 1.0
        m_w = 1.0

    tv_h = (torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2) * m_h).sum()
    tv_w = (torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2) * m_w).sum()
    norm = max((m_h.sum() + m_w.sum()).item(), 1.0) if mask is not None else (b * t * h * w)
    return weight * (tv_h + tv_w) / norm


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

    weight_sum = (weights * mask).sum()
    if weight_sum < 1.0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    mse = weighted_diff.sum() / (weight_sum + 1e-6)
    result = torch.sqrt(mse + epsilon)
    if torch.isnan(result) or torch.isinf(result):
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return result


def centered_masked_rmse_loss(
    pred,
    target,
    mask,
    start_w=1.0,
    end_w=5.0,
    ignore_steps=0,
    ignore_weight=0.0,
    mean_weight=0.1,
    epsilon=1e-6,
):
    """去均值 RMSE：分离空间均值与空间变化，强制模型学习空间 pattern。

    标准 RMSE 的最优解是预测全局平均 → 模型学不到空间变化。
    本 loss 先去掉每个 (time, sample) 的空间均值，在"零均值"场上算 RMSE，
    迫使模型必须预测空间差异才能降低 loss。均值准确性通过一个低权重的辅助 loss 保证。

    Args:
        pred:   (B, T, H, W) 预测值
        target: (B, T, H, W) 目标值
        mask:   (B, T, H, W) 有效区域 mask (1=有效, 0=无效)
        mean_weight: 均值匹配 loss 的权重（默认 0.1）
        其余参数同 weighted_masked_rmse_loss

    Returns:
        spatial_loss + mean_weight * mean_loss
    """
    # 1. 计算每个样本每个时间步的空间加权均值
    # mask_sum: (B, T, 1, 1)
    mask_sum = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)

    pred_mean = (pred * mask).sum(dim=(2, 3), keepdim=True) / mask_sum  # (B, T, 1, 1)
    tgt_mean = (target * mask).sum(dim=(2, 3), keepdim=True) / mask_sum

    # 2. 均值匹配 loss（轻量）：保证模型不被允许随意偏移均值
    mean_sq = ((pred_mean - tgt_mean) ** 2).mean()
    # 用 sqrt 使其与 RMSE 量纲一致
    mean_loss = torch.sqrt(mean_sq + epsilon)

    # 3. 去均值场：pred - mean(pred), target - mean(target)
    pred_centered = pred - pred_mean
    tgt_centered = target - tgt_mean

    # 4. 在去均值场上计算 RMSE（复用加权逻辑）
    spatial_loss = weighted_masked_rmse_loss(
        pred_centered,
        tgt_centered,
        mask,
        start_w=start_w,
        end_w=end_w,
        ignore_steps=ignore_steps,
        ignore_weight=ignore_weight,
        epsilon=epsilon,
    )

    # 5. 总 loss = 空间变化 loss + 轻量均值约束
    total = spatial_loss + mean_weight * mean_loss

    if torch.isnan(total) or torch.isinf(total):
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return total


def smart_background_l1_loss(pred, target, mask, zero_threshold=0.1):
    """
    智能 L1：只惩罚 Ground Truth 本来就是 0 (背景) 的区域
    防止 L1 过大导致模型不敢修正真实偏差
    """
    is_background = (torch.abs(target) < zero_threshold).float()
    bg_mask = mask * is_background
    bg_sum = bg_mask.sum()
    if bg_sum < 1.0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    loss = (torch.abs(pred) * bg_mask).sum() / (bg_sum + 1e-6)
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return loss


def spatial_variance_loss(pred, target, mask, epsilon=1e-6):
    """空间方差匹配 loss：强制模型的预测具有与目标相同的空间变化幅度。

    如果模型只预测常数（var≈0）而目标有显著空间变化（var>0），这个 loss 会很大。
    直接解决"模型学不到空间变化"的核心问题。

    Args:
        pred:   (B, T, H, W) 预测值
        target: (B, T, H, W) 目标值
        mask:   (B, T, H, W) 有效区域 mask

    Returns:
        scalar loss = MSE(pred_spatial_var, target_spatial_var)
    """
    # 每样本每时间步的加权空间方差
    # mask_sum: (B, T, 1, 1)
    mask_sum = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)

    pred_mean = (pred * mask).sum(dim=(2, 3), keepdim=True) / mask_sum
    tgt_mean = (target * mask).sum(dim=(2, 3), keepdim=True) / mask_sum

    # 加权方差 = sum(w * (x - mean)^2) / sum(w)
    pred_var = ((pred - pred_mean) ** 2 * mask).sum(dim=(2, 3)) / mask_sum.squeeze(3).squeeze(2)
    tgt_var = ((target - tgt_mean) ** 2 * mask).sum(dim=(2, 3)) / mask_sum.squeeze(3).squeeze(2)

    # MSE between variances, then sqrt to match RMSE scale
    var_mse = ((pred_var - tgt_var) ** 2).mean()
    var_loss = torch.sqrt(var_mse + epsilon)

    if torch.isnan(var_loss) or torch.isinf(var_loss):
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return var_loss
