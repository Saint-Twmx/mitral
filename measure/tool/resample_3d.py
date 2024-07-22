import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import torch
def uniform_resample(points, n_points):
    points = np.array(points)
    return points[np.linspace(0, len(points) - 1, n_points, dtype=int)]

def douglas_peucker_resample(points, epsilon):
    from rdp import rdp
    return rdp(points, epsilon)

def resample_curve(points, n_points):
    points = np.array(points)
    t = np.linspace(0, 1, len(points))
    t_resampled = np.linspace(0, 1, n_points)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    fx = interp1d(t, x, kind='cubic')
    fy = interp1d(t, y, kind='cubic')
    fz = interp1d(t, z, kind='cubic')

    x_resampled = fx(t_resampled)
    y_resampled = fy(t_resampled)
    z_resampled = fz(t_resampled)

    return torch.tensor(np.stack([x_resampled, y_resampled, z_resampled], axis=-1),
                        dtype=torch.float64)


def resample_curve_new(points, n_points):
    def compute_arc_length(points):
        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        return np.insert(np.cumsum(distances), 0, 0)
    points = np.array(points)
    arc_length = compute_arc_length(points)
    t_resampled = np.linspace(0, arc_length[-1], n_points)

    cs_x = CubicSpline(arc_length, points[:, 0], bc_type='periodic')
    cs_y = CubicSpline(arc_length, points[:, 1], bc_type='periodic')
    cs_z = CubicSpline(arc_length, points[:, 2], bc_type='periodic')

    x_resampled = cs_x(t_resampled)
    y_resampled = cs_y(t_resampled)
    z_resampled = cs_z(t_resampled)

    return torch.tensor(np.stack([x_resampled, y_resampled, z_resampled], axis=-1), dtype=torch.float64)


def upsample_curve(points, upsample_factor):
    if points.shape[0] < 2:
        raise ValueError("Need at least two points for upsampling.")
    upsampled_points = []
    for i in range(points.shape[0] - 1):
        upsampled_points.append(points[i])
        # 在当前点和下一个点之间插入点
        for j in range(upsample_factor - 1):
            new_point = points[i] + (points[i+1] - points[i]) / upsample_factor * (j+1)
            upsampled_points.append(new_point)
    # 添加最后一个点
    upsampled_points.append(points[-1])
    return torch.stack(upsampled_points)

from scipy.interpolate import make_interp_spline
def smooth_points_3d(points, smooth_factor=400, spline_order=3):
    """
    使用B样条对三维点集进行平滑。

    参数:
        points (torch.Tensor): 输入的三维点集，形状应为(N, 3)。
        smooth_factor (int): 平滑后点的数量。
        spline_order (int): B样条曲线的阶数。

    返回:
        torch.Tensor: 平滑后的三维点集。
    """
    # 确保输入是numpy数组
    if isinstance(points, torch.Tensor):
        points = points.numpy()

    # 生成参数化变量t
    t = np.linspace(0, 1, len(points))

    # 创建B样条曲线
    spl_x = make_interp_spline(t, points[:, 0], k=spline_order)
    spl_y = make_interp_spline(t, points[:, 1], k=spline_order)
    spl_z = make_interp_spline(t, points[:, 2], k=spline_order)

    # 生成更多的t值用于平滑曲线
    smooth_t = np.linspace(0, 1, smooth_factor)
    smooth_points = np.vstack((spl_x(smooth_t), spl_y(smooth_t), spl_z(smooth_t))).T

    # 将平滑后的点转换为Tensor
    return torch.tensor(smooth_points, dtype=torch.float32)