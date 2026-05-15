import os
import yaml
import torch


def _convert_types(obj):
    if isinstance(obj, dict):
        return {k: _convert_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_types(x) for x in obj]
    if isinstance(obj, str):
        try:
            if any(c in obj for c in [".", "e", "E"]):
                return float(obj)
            if obj.isdigit():
                return int(obj)
        except ValueError:
            pass
    return obj


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _convert_types(cfg)


def mkdirs(path):
    os.makedirs(path, exist_ok=True)
    print(f"[mkdir] Directory ready: {path}")
    return path


def chamfer_distance(pc1, pc2):
    """
    Compute Chamfer Distance between two point clouds.

    Args:
        pc1: Tensor of shape [B, N, 3].
        pc2: Tensor of shape [B, M, 3].

    Returns:
        Scalar Chamfer Distance.
    """
    dist = torch.cdist(pc1, pc2)  # [B, N, M]
    min_dist_pc1_to_pc2 = torch.min(dist, dim=2).values  # [B, N]
    min_dist_pc2_to_pc1 = torch.min(dist, dim=1).values  # [B, M]
    return torch.mean(min_dist_pc1_to_pc2) + torch.mean(min_dist_pc2_to_pc1)


def earth_movers_distance(pc1, pc2):
    """
    Compute a simplified Earth Mover's Distance.

    Args:
        pc1: Tensor of shape [B, N, 3].
        pc2: Tensor of shape [B, N, 3].

    Returns:
        Tensor of shape [B].
    """
    return torch.mean(torch.abs(pc1 - pc2), dim=(1, 2))


def pearson_correlation(x, y, eps=1e-8):
    """
    Compute Pearson correlation coefficient between two tensors.

    Args:
        x: Input tensor.
        y: Input tensor with the same shape as x.
        eps: Small value for numerical stability.

    Returns:
        Scalar Pearson correlation coefficient.
    """
    x_centered = x - x.mean()
    y_centered = y - y.mean()

    covariance = (x_centered * y_centered).sum()
    x_std = torch.sqrt((x_centered ** 2).sum())
    y_std = torch.sqrt((y_centered ** 2).sum())

    return covariance / (x_std * y_std + eps)