from measure.mitral_bestplane import project_points_onto_plane_gpu
from measure.tool.concaveHull import  get_outer_contour_3d
import numpy as np
import torch
from measure.tool.coordinate_calculate import convert_to_physical_coordinates
from measure.tool.coordinate_calculate import calculate_perimeter, calculate_area_of_3d_polygon_with_centroid
from measure.tool.spendtime import log_time
from measure.tool.resample_3d import resample_curve,upsample_curve
from measure.mitral_bestplane import binary_erosion_3d
from measure.tool.curvature import curvature_curve

def calculate_points_distance_torch(
    pts1: torch.Tensor, pts2: torch.Tensor
) -> torch.Tensor:
    pts1 = pts1.to(dtype=torch.float64)
    pts2 = pts2.to(dtype=torch.float64)
    return torch.sqrt_(
        torch.sum(torch.pow(pts1[:, None, :] - pts2[None, :, :], 2), dim=-1)
    )

def uniform_resample(points, tolerance=0.01, max_iterations=100):
    def distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
    n = len(points)
    target_length = np.sum([distance(points[i], points[(i + 1) % n]) for i in range(n)]) / n
    for _ in range(max_iterations):
        distances = [distance(points[i], points[(i + 1) % n]) for i in range(n)]
        max_deviation = max([abs(d - target_length) for d in distances])
        if max_deviation < tolerance:
            break
        max_index = distances.index(max(distances, key=lambda x: abs(x - target_length)))
        if distances[max_index] > target_length:
            move_factor = (distances[max_index] - target_length) / distances[max_index]
            mid_point = (points[max_index] + points[(max_index + 1) % n]) / 2
            points[max_index] = points[max_index] + (mid_point - points[max_index]) * move_factor
            points[(max_index + 1) % n] = points[(max_index + 1) % n] + (
                        mid_point - points[(max_index + 1) % n]) * move_factor
    return torch.tensor(points)

def filter_points_by_distance(points: torch.Tensor, point: torch.Tensor, N: float) -> torch.Tensor:
    # 确保 point 是一个行向量，即一维张量
    if point.dim() == 1:
        point = point.unsqueeze(0)
    differences = points - point
    squared_distances = torch.sum(differences ** 2, dim=1)
    distances = torch.sqrt(squared_distances)
    filtered_points = points[distances > N]
    return filtered_points

def sort_circle_points(points):
    start_point = points[0]
    sorted_points = [start_point]
    # 从剩余的点中选择下一个最接近的点，并添加到已排序的点列表中
    remaining_points = points[1:].clone()
    while remaining_points.shape[0] > 0:
        distances = torch.norm(remaining_points - sorted_points[-1], dim=1)
        next_point_index = torch.argmin(distances)
        next_point = remaining_points[next_point_index]
        sorted_points.append(next_point)
        remaining_points = torch.cat((remaining_points[:next_point_index], remaining_points[next_point_index+1:]))
    # 将列表转换为张量
    sorted_points = torch.stack(sorted_points)
    return sorted_points


def split_into_sublists(lst):
    sublists = []
    sublist = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] - lst[i-1] == 1:
            sublist.append(lst[i])
        else:
            sublists.append(sublist)
            sublist = [lst[i]]
    sublists.append(sublist)
    return sublists

@log_time
def mit_annulus_perimeter_area(ori_pred, head, threeD_plane, best_plane, measure):
    valve = ori_pred.copy()
    valve[valve > 2] = 0
    valve[valve > 0] = 1
    boundary = binary_erosion_3d(valve, iterations=2)
    shell = torch.logical_xor(torch.from_numpy(valve), boundary)
    mitral_point = torch.nonzero(shell == 1)

    projection_point_proj = project_points_onto_plane_gpu(mitral_point, best_plane, True)
    hull_point_proj, _ = get_outer_contour_3d(np.array(projection_point_proj.cpu()), np.array(best_plane), types=True)

    projection_point_3d = project_points_onto_plane_gpu(mitral_point, threeD_plane, True)
    hull_point_3d, hull_point_3d_index = get_outer_contour_3d(np.array(projection_point_3d.cpu()), np.array(threeD_plane),types=True)

    hull_point_proj = resample_curve(hull_point_proj.cpu(), 36)
    hull_point_3d = resample_curve(hull_point_3d.cpu(), 36)   # smooth_points_3d

    mitral_hull_3d_dis = calculate_points_distance_torch(hull_point_3d.cuda(), mitral_point.cuda())
    _, hull_3d_index = torch.min(mitral_hull_3d_dis, dim=1)
    mitral_hull_point = mitral_point[hull_3d_index.cpu()]

    hull_point_3d = resample_curve(hull_point_proj.cpu(), 72)
    mitral_hull_point = resample_curve(mitral_hull_point, 72)   # smooth_points_3d

    hull_point_3d = curvature_curve(hull_point_3d, model='example')
    mitral_hull_point = curvature_curve(mitral_hull_point, model='example')
    hull_point_3d = resample_curve(hull_point_3d.cpu(), 36)
    mitral_hull_point = resample_curve(mitral_hull_point, 36)

    measure["hull_point_3d"] = hull_point_3d
    measure["mitral_hull_point"] = mitral_hull_point
    measure["hull_point_3d_resample"] = upsample_curve(hull_point_3d, 5)
    measure["mitral_hull_point_resample"] = upsample_curve(mitral_hull_point, 5)


    physical_mitral_points = convert_to_physical_coordinates(mitral_hull_point, head['spacing'])
    physical_hull_points = convert_to_physical_coordinates(hull_point_3d, head['spacing'])

    mitral_perimeter = calculate_perimeter(physical_mitral_points, types=3)
    mitral_proj_perimeter= calculate_perimeter(physical_hull_points, types=3)

    mitral_area = calculate_area_of_3d_polygon_with_centroid(physical_mitral_points)
    mitral_proj_area = calculate_area_of_3d_polygon_with_centroid(physical_hull_points)

    measure["mitral_perimeter"] = mitral_perimeter
    measure["mitral_proj_perimeter"] = mitral_proj_perimeter
    measure["mitral_area"] = mitral_area
    measure["mitral_proj_area"] = mitral_proj_area
