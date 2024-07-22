import torch
import numpy as np
from measure.tool.readdicom import get_info_with_sitk_nrrd, handle_save_array
from measure.tool.coordinate_calculate import fit_plane_pca
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial import cKDTree
from measure.tool.spendtime import log_time

def calculate_points_plane_distance(
        pts: torch.Tensor, plane: torch.Tensor
        ):
    return (torch.sum(plane[0:3] * pts, dim=1) + plane[3]) / torch.sqrt(
        torch.sum(torch.pow(plane[0:3], 2))
    )

def binary_erosion_3d(input_matrix, iterations=1):
    eroded_matrix = binary_erosion(input_matrix, iterations=iterations)
    return torch.tensor(eroded_matrix, dtype=torch.bool)

def binary_dilation_3d(input_matrix, iterations=1):
    eroded_matrix = binary_dilation(input_matrix, iterations=iterations)
    return torch.tensor(eroded_matrix, dtype=torch.bool)

def project_points_onto_plane_gpu(points, plane, integer=False):
    points = points.to(torch.float32).cuda()
    plane = plane.to(torch.float32).cuda()
    A, B, C, D = plane
    normal_vector = torch.tensor([A, B, C], dtype=torch.float32, device='cuda')
    denominator = torch.dot(normal_vector, normal_vector)
    t = -(torch.matmul(points, normal_vector) + D) / denominator
    projected_points = points + t.unsqueeze(1) * normal_vector
    if integer:
        return torch.round(projected_points)
    return projected_points

def check_plane_direction(pointA, pointB, plane):
    # 给定中心线AB两点，判定平面方程，指定面“朝下”
    vector_AB = pointA - pointB
    normal_vector = plane[:3]
    dot_product = torch.dot(vector_AB, normal_vector)
    norm_AB = torch.norm(vector_AB)
    norm_normal = torch.norm(normal_vector)
    cos_theta = dot_product / (norm_AB * norm_normal)
    theta = torch.acos(cos_theta)
    theta_degrees = theta * (180.0 / torch.pi)
    if theta_degrees<90:
        plane= -plane
    return plane

@log_time
def mit_bestplane_new(ori_pred, centerline, measure):
    mitral_point = torch.nonzero(torch.from_numpy((ori_pred == 1)|(ori_pred == 2))).type(torch.float32)
    best_plane = fit_plane_pca(mitral_point)
    best_plane = check_plane_direction(centerline[0], centerline[-1], best_plane)

    ad_co_bp_dis = calculate_points_plane_distance(mitral_point.cpu(), best_plane.cpu())
    tmp_num = 0
    print(f"平面 MAX:{torch.max(ad_co_bp_dis)}, best_plane:{best_plane}")
    while True:  #
        if torch.max(ad_co_bp_dis) < np.sqrt(6) or tmp_num > 10 or ad_co_bp_dis[ad_co_bp_dis > 0].mean() <= np.sqrt(3):
            break
        adjacent_coords = mitral_point[(ad_co_bp_dis > 1-np.sqrt(3))]
        best_plane = fit_plane_pca(adjacent_coords)
        best_plane = check_plane_direction(centerline[0], centerline[-1], best_plane)
        ad_co_bp_dis = calculate_points_plane_distance(mitral_point.cpu(), best_plane.cpu())
        tmp_num+=1
        # print(f"平面 max {tmp_num} ：{torch.max(ad_co_bp_dis)}, best_plane:{best_plane}")
    measure["mitral_point"] = mitral_point

    tmp_value = 0
    tmp_plane = best_plane.clone()
    while True:
        tmp_plane[-1] -= 0.5
        plane_dis = calculate_points_plane_distance(mitral_point, tmp_plane)
        values_below_threshold = (plane_dis < 0.1)
        count = torch.sum(values_below_threshold).item()
        if count / len(mitral_point) > 0.9995 or tmp_value>100:
            print(f"tmp_plane:{tmp_plane}, best_plane:{best_plane}")
            tmp_plane = check_plane_direction(centerline[0], centerline[-1], tmp_plane)
            return tmp_plane, best_plane
        tmp_value += 1
    return np.array([]), np.array([])



if __name__ == "__main__":
    # test
    paths = r"/mitral"
    nrrdpath = r"2053_75%_2.seg.nrrd"
    import os
    from measure.mitral_centerline import mit_centerline
    from measure.mitral_planes import mit_planes

    ori_pred, head = get_info_with_sitk_nrrd(os.path.join(paths, nrrdpath))

    centerline = mit_centerline(ori_pred, types="2")

    plane_variants_normal, mitral_point = mit_planes(ori_pred, centerline, types="2")

    projection_point_undetermined = mit_bestplane(plane_variants_normal, mitral_point, ori_pred, head)


