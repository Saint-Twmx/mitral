from scipy.spatial import ConvexHull
import torch

def project_to_2d(points, plane_normal):
    # 根据平面法线选择投影方法
    if abs(plane_normal[0]) > abs(plane_normal[1]) and abs(plane_normal[0]) > abs(plane_normal[2]):
        return points[:, 1:]
    elif abs(plane_normal[1]) > abs(plane_normal[2]):
        return points[:, [0, 2]]
    else:
        return points[:, :2]


def get_outer_contour_3d(points_3d, plane_normal, types = False, _2d = True): # 这个玩意只能算凸包 并且点少  但是快
    if _2d:
        points_2d = project_to_2d(points_3d, plane_normal)
        hull = ConvexHull(points_2d)
        hull_indices = hull.vertices
    else:
        hull = ConvexHull(points_3d)
        hull_indices = hull.vertices
    outer_contour_3d = points_3d[hull_indices]
    if types:
        return torch.tensor(outer_contour_3d, dtype=torch.float32),\
               torch.tensor(hull_indices, dtype=torch.float32)
    return outer_contour_3d,hull_indices
