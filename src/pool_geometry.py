"""Tank geometry for the salvage-ROV pool.

Analytic helpers shared by the model and the demo: the water surface and the floor
are sloped, so wall distances are computed in closed form rather than tabulated.
Extracted from the training data loader so that the demo runs without a dataset.
"""
import torch

POOL_X_MIN, POOL_X_MAX = -8.25, 8.25
POOL_Y_MIN, POOL_Y_MAX = -28.0, -9.5
POOL_WARN_MARGIN = 0.5
CONTACT_THICKNESS = 0.5
CONTACT_XI = max(4 * CONTACT_THICKNESS, 0.1)


def pool_z_top(y_val):
    """水面高度 (Z上界)"""
    return 0.5 + (float(y_val) + 9.5) / 18.5


def pool_z_bottom(x_val, y_val):
    """水底深度 (Z下界)"""
    abs_x = abs(float(x_val))
    if abs_x <= 6.75:
        return -9.5
    t = (abs_x - 6.75) / 1.5
    return -9.5 + t * (pool_z_top(y_val) + 9.5)


def compute_wall_distances(pos_xyz):
    """
    计算6面墙的距离和法线 (NeRD式接触量)

    Args:
        pos_xyz: (..., 3) 位置 [x, y, z]

    Returns:
        dist: (..., 6) 到6面墙的距离 [x_min, x_max, y_min, y_max, z_top, z_bot]
        normals: (..., 6, 3) 6面墙的内法线
    """
    x, y, z = pos_xyz[..., 0], pos_xyz[..., 1], pos_xyz[..., 2]

    d_x_min = x - POOL_X_MIN
    d_x_max = POOL_X_MAX - x
    d_y_min = y - POOL_Y_MIN
    d_y_max = POOL_Y_MAX - y

    z_top = 0.5 + (y + 9.5) / 18.5
    abs_x = x.abs()
    z_bot = torch.where(
        abs_x <= 6.75,
        torch.full_like(z, -9.5),
        -9.5 + ((abs_x - 6.75) / 1.5) * (z_top + 9.5)
    )
    d_z_top = z_top - z
    d_z_bot = z - z_bot

    dist = torch.stack([d_x_min, d_x_max, d_y_min, d_y_max, d_z_top, d_z_bot], dim=-1)

    # 内法线 (指向水池内部)
    normals = torch.zeros(*pos_xyz.shape[:-1], 6, 3, device=pos_xyz.device, dtype=pos_xyz.dtype)
    normals[..., 0, 0] = 1.0   # x_min → +x
    normals[..., 1, 0] = -1.0  # x_max → -x
    normals[..., 2, 1] = 1.0   # y_min → +y
    normals[..., 3, 1] = -1.0  # y_max → -y
    normals[..., 4, 2] = -1.0  # z_top → -z (向下)
    normals[..., 5, 2] = 1.0   # z_bot → +z (向上)

    return dist, normals
