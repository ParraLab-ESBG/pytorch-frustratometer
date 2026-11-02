"""Shared utility functions for all frustration modes (no circular deps)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def check_size(t, max_len=400):
    t = torch.atleast_1d(t)
    if t.size(0) >= max_len:
        return t[:max_len]
    else:
        return torch.cat((
            t, torch.zeros((max_len - t.size(0), *list(t.size()[1:])), dtype=t.dtype)
        ))

def apply_random_se3(coords_in, atom_mask=None, translation_scale=1.0, random_rot=None, return_matrix=False):
    coords_mean = coords_in[:, 1:2].mean(-3, keepdim=True)
    coords_in -= coords_mean
    if random_rot is None:
        random_rot, _ = torch.linalg.qr(torch.randn(3, 3))
    coords_in = coords_in @ random_rot.type(coords_in.dtype)
    random_trans = torch.randn_like(coords_mean) * translation_scale
    coords_in += random_trans
    if atom_mask is not None:
        coords_in = coords_in * atom_mask[..., None]
    if return_matrix:
        return coords_in, random_rot
    return coords_in


def index_to_pseudo_logits(index, threshold, temp=0.5):
    """
    Convert a scalar frustration index tensor to pseudo class logits using
    the same sigmoid soft-thresholding as consistency_loss.

    Args:
        index:     any shape (float tensor)
        threshold: 1-D tensor  [low_cut, high_cut]  (e.g. [-1, 0.55])
        temp:      softness of the sigmoid

    Returns:
        log-probabilities of shape (*index.shape, 3)
        class order: [highly, neutral, minimally]
    """
    thr = threshold.to(index.device)
    p_high    = torch.sigmoid(-(index - thr[0]) / temp)
    p_minimal = torch.sigmoid( (index - thr[1]) / temp)
    p_neutral = (1.0 - p_high - p_minimal).clamp(min=0.0)
    probs = torch.stack([p_high, p_neutral, p_minimal], dim=-1)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return probs.log().clamp(min=-100.0)


def compute_frustration_metrics(pred, true, mask, threshold):
    """
    Compute MSE, MAE, Pearson R, and class accuracy for a frustration index.

    Args:
        pred:      [B, L] or [B, E] predicted frustration index
        true:      same shape
        mask:      same shape (bool)
        threshold: 1-D class-boundary tensor

    Returns dict with keys: n, mse, mae, pearson_r, class_accuracy
    """
    pred_m = pred[mask.bool()].float()
    true_m = true[mask.bool()].float()
    n = pred_m.numel()
    if n == 0:
        nan = float("nan")
        return {"n": 0, "mse": nan, "mae": nan, "pearson_r": nan, "class_accuracy": nan}

    diff = pred_m - true_m
    mse  = (diff ** 2).mean().item()
    mae  = diff.abs().mean().item()
    vx   = pred_m - pred_m.mean()
    vy   = true_m - true_m.mean()
    r    = ((vx * vy).sum() / (vx.norm() * vy.norm() + 1e-8)).item()

    import scipy.stats
    spearmean_r = scipy.stats.spearmanr(pred_m.cpu(), true_m.cpu()).statistic

    cosine_sim = F.cosine_similarity(pred_m.unsqueeze(0), true_m.unsqueeze(0)).item()

    thr      = threshold.to(pred.device)
    pred_cls = torch.searchsorted(thr, pred_m)
    true_cls = torch.searchsorted(thr, true_m)
    class_acc = (pred_cls == true_cls).float().mean().item()

    pred_std  = pred_m.std().item()
    true_std  = true_m.std().item()
    std_ratio = (pred_std / true_std) if true_std > 1e-8 else float("nan")

    n_cls = len(thr) + 1
    pred_frac = [(pred_cls == c).float().mean().item() for c in range(n_cls)]
    true_frac = [(true_cls == c).float().mean().item() for c in range(n_cls)]

    return {
        "n": n, "mse": mse, "mae": mae, "pearson_r": r, "spearman_r": spearmean_r,
        "cosine_similarity": cosine_sim,
        "class_accuracy": class_acc,
        "pred_std": pred_std, "true_std": true_std, "std_ratio": std_ratio,
        "pred_frac_cls0": pred_frac[0], "pred_frac_cls1": pred_frac[1], "pred_frac_cls2": pred_frac[2],
        "true_frac_cls0": true_frac[0], "true_frac_cls1": true_frac[1], "true_frac_cls2": true_frac[2],
    }


def frustration_loss(
    mode,
    pred_frustration_index,
    pred_frustration_class,
    true_frustration_index,
    seq_mask,
    true_frustration_class=None,
    welltype=None,
    reduce=True,
    pair=False,
    stage="train",
    index_loss_type="huber",
    label_smoothing=0.0,
    use_ordinal=False,
    use_focal=False,
    focal_gamma=2.0,
):
    try:
        from frustrai_fold.utils.loss import frustration_loss as _ff_loss
        return _ff_loss(
            mode, pred_frustration_index, pred_frustration_class,
            true_frustration_index, seq_mask,
            true_frustration_class=true_frustration_class,
            welltype=welltype, reduce=reduce, pair=pair, stage=stage,
            index_loss_type=index_loss_type, label_smoothing=label_smoothing,
            use_ordinal=use_ordinal, use_focal=use_focal, focal_gamma=focal_gamma,
        )
    except ImportError:
        nan = torch.tensor(float("nan"), device=pred_frustration_index.device)
        return nan, nan
