import torch
import numpy as np
from measure.mitral_bestplane import project_points_onto_plane_gpu, check_plane_direction
from measure.mitral_planes import calculate_points_plane_distance, calculate_points_distance_torch
from measure.tool.concaveHull import compute_concave_hull_perimeter
from measure.tool.resample_3d import resample_curve, resample_curve_new
from measure.tool.coordinate_calculate import convert_to_physical_coordinates, fit_plane_pca
from measure.tool.show import show_plane
from measure.tool.readdicom import get_info_with_sitk_nrrd, handle_save_array
from measure.mitral_six_subarea import mit_six_subarea
def find_perpendicular_planes(point, plane):
    # 给一个点（经过平面） 和一个平面，得到另外两个正交的平面方程
    def calculate_plane_equations(point, normal_vector, device):
        x0, y0, z0 = point[0]
        D = -(normal_vector[0] * x0 + normal_vector[1] * y0 + normal_vector[2] * z0)
        D = D.to(device)
        return torch.cat((normal_vector, torch.tensor([D]).to(device)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    point = point.to(device)
    plane = plane.to(device)
    normal_vector = plane[:3]
    vector1 = torch.tensor([1, 0, -normal_vector[0] / normal_vector[2]], device=device)
    vector2 = torch.cross(normal_vector, vector1)
    plane1_params = calculate_plane_equations(point, vector1, device)
    plane2_params = calculate_plane_equations(point, vector2, device)
    return plane1_params, plane2_params

def calculate_parallel_plane(plane, point):
    # 给一个平面与一个点，计算经过该点并且和原平面平行的平面
    normal_vector = plane[:3]
    x0, y0, z0 = point
    numerator = torch.abs(normal_vector[0] * x0 + normal_vector[1] * y0 + normal_vector[2] * z0 + plane[3])
    denominator = torch.sqrt(torch.sum(normal_vector ** 2))
    distance = numerator / denominator
    new_D = plane[3] - numerator * torch.sign(normal_vector[0] * x0 + normal_vector[1] * y0 + normal_vector[2] * z0 + plane[3])
    return torch.cat((normal_vector, new_D.unsqueeze(0)))


def leaflet_curve(points, plane, rate=0.5, num=40, anchor=None):
    def split_and_rotate_list(lst, index_A, index_B):
        if index_A > index_B:
            index_A, index_B = index_B, index_A
        if index_B - index_A>=20:
            part1 = lst[index_A:index_B]
            part2 = torch.cat((lst[index_B:], lst[:index_A]), dim=0)
        else:
            part1 = lst[index_A:index_B + 1]
            part2 = torch.cat((lst[index_B + 1:], lst[:index_A]), dim=0)
        return part1, part2
    points_dis = calculate_points_plane_distance(points, plane)
    points_threshold = (torch.abs(points_dis) < 1)
    # anchor 更改使用逻辑，如果  points_dis 【-1,0】【0,1】数量严重不一致，说明当前平面plane，偏了，需要像 anchor靠近

    points_points = points[points_threshold]
    points_points_proj = project_points_onto_plane_gpu(points_points, plane.cpu(), True)
    _, new_points, _ = compute_concave_hull_perimeter(points_points_proj.cpu(), rate)
    new_points = resample_curve(new_points[:-1], num)
    # from measure.tool.curvature import curvature_curve
    # curvature_curve(new_points, model='example')

    # 20240418 更换前后两个点索引逻辑
    dis = calculate_points_distance_torch(new_points, new_points)
    tmp_v, tmp_i = torch.max(dis, dim=0)
    _, max_index = torch.max(tmp_v,dim=0)
    min_index = tmp_i[max_index]
    if abs(max_index - min_index)>3 and abs(max_index - min_index)<=len(new_points)/2:
        part1, part2 = split_and_rotate_list(new_points,max_index,min_index)
        if len(part1) != len(part2):
            len1, len2 = part1.shape[0], part2.shape[0]
            if len1 < len2: part1 = resample_curve(part1, len2)
            elif len2 < len1: part2 = resample_curve(part2, len1)
        return resample_curve((part1 + torch.flip(part2, [0])) / 2, 5)
    else:
        # 重述这组数
        star_point = new_points[max_index]
        distances = torch.norm(new_points - star_point, dim=1)
        sorted_indices = distances.argsort()
        sorted_new_points = new_points[sorted_indices]
        return resample_curve(sorted_new_points, 5)


def plane_equation_through_midpoint(A, B, spacing=[1,1,1]):
    # 俩点，得到经过中点的 垂直平面  # spacing矫正方向
    spacing = torch.tensor(spacing, dtype=torch.float32)
    AB = (B - A) * spacing
    M = (A + B) / 2
    a, b, c = AB[0], AB[1], AB[2]
    d = -(a * M[0] + b * M[1] + c * M[2])
    return torch.tensor([a, b, c, d], dtype=A.dtype)


def find_perpendicular_plane(points, best_plane, spacing=[1,1,1]):
    spacing = torch.tensor(spacing, dtype=torch.float32)
    adjusted_best_normal = best_plane[:3] / spacing     # 调整 best_plane 的法向量
    v = points[1] - points[0]    # 计算并调整方向向量
    adjusted_v = v  # 是否需要调整？
    n_new = torch.cross(adjusted_best_normal, adjusted_v)    # 计算叉积
    n_new = n_new  # 是否需要调整？
    n_new = n_new / torch.norm(n_new)# 归一化 n_new
    d = -torch.dot(n_new, points[0])# 计算 d 参数 (使平面通过 points[0])
    new_plane = torch.cat([n_new, d.unsqueeze(0)])
    return new_plane

def find_perpendicular_plane_new(points, best_plane, spacing=[1,1,1]):
    # 脱胎与find_perpendicular_plane
    # 经过points的中点，以一种奇怪的方式正交了，但是并一定通过points两个点
    spacing = torch.tensor(spacing, dtype=torch.float32)
    adjusted_best_normal = best_plane[:3] / spacing
    adjusted_v = points[1] - points[0]
    n_new = torch.cross(adjusted_best_normal, adjusted_v)
    n_new = n_new * spacing
    n_new = n_new / torch.norm(n_new)
    d = -torch.dot(n_new, torch.mean(points,dim=0))
    new_plane = torch.cat([n_new, d.unsqueeze(0)])
    return new_plane

def claw(A1_points, ori_pred, best_plane, A123, type='A'):
    # tensor1 = A1_points[-1].type(torch.float)
    tensor2 = torch.mean(torch.nonzero(torch.from_numpy((ori_pred == 1) | (ori_pred == 2))).type(torch.float), dim=0)
    tensor1 = A1_points[torch.max(calculate_points_distance_torch(A1_points, tensor2.unsqueeze(0)),dim=0)[1]]

    if type=='A':
        # 结合2叶子中点，拉伸
        tmp = torch.mean(torch.nonzero(torch.from_numpy((ori_pred == 2))).type(torch.float), dim=0)
        tensor2 = (2 * tmp + tensor2) / 3
    elif type=='P':
        # 结合1叶子中点，拉伸
        tmp = torch.mean(torch.nonzero(torch.from_numpy((ori_pred == 1))).type(torch.float), dim=0)
        tensor2 = (2 * tmp + tensor2) / 3

    target1_plane_z = find_perpendicular_plane(
        torch.cat((tensor1.type(torch.float), tensor2.unsqueeze(0)), dim=0),
        best_plane.type(torch.float)
    )
    return leaflet_curve(A123, target1_plane_z.cpu(), anchor=best_plane)

def calculated_value(measure, name, points, head):
    measure[f"{name}_points"] = points
    points_phy = convert_to_physical_coordinates(points.cpu(), head['spacing'])
    measure[f"{name}_points_line_dis"] = np.linalg.norm(points_phy[0] - points_phy[-1])
    measure[f"{name}_points_curve_dis"] = sum([np.linalg.norm(points_phy[i] - points_phy[i+1]) for i in range(len(points_phy)-1)])

def move_plane(A1points, A1_leaflet_plane_dis, A1_leaflet_plane, targetA2_plane):
    try:
        L1 = len(A1points[(A1_leaflet_plane_dis > 0) & (A1_leaflet_plane_dis < 2.5)])
        L2 = len(A1points[(A1_leaflet_plane_dis > -2.5) & (A1_leaflet_plane_dis < 0)])
        monitor_plane = A1_leaflet_plane.clone()
        monitor_ratio = L1 / L2
        while L1 != 0 and L2 != 0 and (L1/L2 > 1.5 or L1/L2 < 0.5):
            # 循环平移该平面
            A1_leaflet_plane[-1] = A1_leaflet_plane[-1] + (targetA2_plane[-1] - A1_leaflet_plane[-1]) * 0.025
            A1_leaflet_plane_dis = calculate_points_plane_distance(A1points, A1_leaflet_plane)
            L1 = len(A1points[(A1_leaflet_plane_dis > 0) & (A1_leaflet_plane_dis < 2.5)])
            L2 = len(A1points[(A1_leaflet_plane_dis > -2.5) & (A1_leaflet_plane_dis < 0)])
            if L1 == 0 or L2 == 0 or abs(L1/L2 - 1) > abs(monitor_ratio - 1): # 正在变差
                return monitor_plane
            else:
                monitor_ratio = L1 / L2
                monitor_plane = A1_leaflet_plane.clone()
    except Exception as e:
        print(f"warning : {e}")
    return A1_leaflet_plane


def mit_leaflets_length(ori_pred, head, best_plane, measure, types="2", subarea6=False):

    cc_points = measure["mitral_cc_points"]
    # cc_centre_proj_point = project_points_onto_plane_gpu(torch.mean(measure["mitral_hull_point"],dim=0).unsqueeze(0),
    #                                                      best_plane)
    # # 使用 best_plane以及一个中心点确定候选俩平面
    # plane1_params, plane2_params = find_perpendicular_planes(cc_centre_proj_point, best_plane)
    # # 主瓣点集
    # aortic_points = torch.nonzero(torch.from_numpy((ori_pred == 5) | (ori_pred == 6) | (ori_pred == 7)))
    # plane1_dis = calculate_points_plane_distance(aortic_points, plane1_params.cpu())
    # plane2_dis = calculate_points_plane_distance(aortic_points, plane2_params.cpu())
    # if torch.min(torch.abs(plane1_dis)) < torch.min(torch.abs(plane2_dis)):
    #     target2_plane = plane1_params
    # else:
    #     target2_plane = plane2_params

    target2_plane = find_perpendicular_plane(measure["mitral_ap_points"].type(torch.float), best_plane.type(torch.float))

    # target1_plane：A1  P1    target2_plane:A2  P2   target3_plane: A3  P3
    # 右心房点集
    yf_points = torch.nonzero(torch.from_numpy((ori_pred == 9)))
    yf_cc_dis = calculate_points_distance_torch(cc_points, yf_points)
    if torch.min(yf_cc_dis[0]) > torch.min(yf_cc_dis[1]):
        one_third_point = cc_points[0] + (cc_points[1] - cc_points[0]) / 4
        two_third_point = cc_points[0] + 3 * (cc_points[1] - cc_points[0]) / 4
    else:
        two_third_point = cc_points[0] + (cc_points[1] - cc_points[0]) / 4
        one_third_point = cc_points[0] + 3 * (cc_points[1] - cc_points[0]) / 4

    target1_plane = calculate_parallel_plane(target2_plane, one_third_point)
    target3_plane = calculate_parallel_plane(target2_plane, two_third_point)

    A123 = torch.nonzero(torch.from_numpy((ori_pred == 1)))
    P123 = torch.nonzero(torch.from_numpy((ori_pred == 2)))

    A123_t1 = A123[torch.abs(calculate_points_plane_distance(A123, target1_plane)) < 30]
    A123_t2 = A123[torch.abs(calculate_points_plane_distance(A123, target2_plane)) < 30]
    A123_t3 = A123[torch.abs(calculate_points_plane_distance(A123, target3_plane)) < 30]
    P123_t1 = P123[torch.abs(calculate_points_plane_distance(P123, target1_plane)) < 30]
    P123_t2 = P123[torch.abs(calculate_points_plane_distance(P123, target2_plane)) < 30]
    P123_t3 = P123[torch.abs(calculate_points_plane_distance(P123, target3_plane)) < 30]

    A1_points = leaflet_curve(A123_t1, target1_plane.cpu(), anchor = best_plane)
    A2_points = leaflet_curve(A123_t2, target2_plane.cpu(), anchor = best_plane)
    A3_points = leaflet_curve(A123_t3, target3_plane.cpu(), anchor = best_plane)
    P1_points = leaflet_curve(P123_t1, target1_plane.cpu(), anchor = best_plane)
    P2_points = leaflet_curve(P123_t2, target2_plane.cpu(), anchor = best_plane)
    P3_points = leaflet_curve(P123_t3, target3_plane.cpu(), anchor = best_plane)


    calculated_value(measure, "A1", A1_points, head)
    calculated_value(measure, "A2", A2_points, head)
    calculated_value(measure, "A3", A3_points, head)
    calculated_value(measure, "P1", P1_points, head)
    calculated_value(measure, "P2", P2_points, head)
    calculated_value(measure, "P3", P3_points, head)

    return

