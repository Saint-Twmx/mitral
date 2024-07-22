from measure.tool.spendtime import log_time
import torch
import numpy as np
from measure.tool.coordinate_calculate import convert_to_physical_coordinates
from measure.mitral_annulus import calculate_points_distance_torch
from measure.mitral_bestplane import calculate_points_plane_distance
from measure.mitral_bestplane import check_plane_direction
from measure.tool.resample_3d import upsample_curve
from measure.mitral_bestplane import project_points_onto_plane_gpu
import copy
def find_perpendicular_plane(points, best_plane, spacing=[1,1,1]):
    spacing = torch.tensor(spacing, dtype=torch.float32)
    adjusted_best_normal = best_plane[:3] / spacing
    v = points[1] - points[0]
    adjusted_v = v
    n_new = torch.cross(adjusted_best_normal, adjusted_v)
    n_new = n_new
    n_new = n_new / torch.norm(n_new)
    d = -torch.dot(n_new, points[0])
    new_plane = torch.cat([n_new, d.unsqueeze(0)])
    return new_plane

def find_perpendicular_plane_new(points, best_plane, spacing=[1,1,1]):
    spacing = torch.tensor(spacing, dtype=torch.float32)
    adjusted_best_normal = best_plane[:3] / spacing
    adjusted_v = points[1] - points[0]
    n_new = torch.cross(adjusted_best_normal, adjusted_v)
    n_new = n_new * spacing
    n_new = n_new / torch.norm(n_new)
    d = -torch.dot(n_new, torch.mean(points,dim=0))
    new_plane = torch.cat([n_new, d.unsqueeze(0)])
    return new_plane

def interpolate_points(points, num):
    if points.shape != (2, 3):
        raise ValueError("Points should be a 2x3 tensor")
    pointA, pointB = points
    step = (pointB - pointA) / (num + 1)
    interpolated_points = [pointA + step * i for i in range(1, num + 1)]
    return torch.stack([pointA] + interpolated_points + [pointB])

def distance_to_line(pointA, pointB, points):
    direction_vector = pointB - pointA
    distances = []
    for point in points:
        vector_AP = point - pointA
        cross_product = torch.cross(vector_AP, direction_vector)
        cross_product_length = torch.linalg.norm(cross_product)
        direction_length = torch.linalg.norm(direction_vector)
        distance = cross_product_length / direction_length
        distances.append(distance)
    return distances

def distance_to_line_gpu(pointA, pointB, points):
    pointA = pointA.to('cuda')
    pointB = pointB.to('cuda')
    points = points.to('cuda')
    # 计算直线方向向量
    direction_vector = pointB - pointA
    direction_length = torch.linalg.norm(direction_vector)
    vector_AP = points - pointA[None, :]
    cross_products = torch.cross(vector_AP, direction_vector.expand(points.size(0), -1))
    cross_product_lengths = torch.linalg.norm(cross_products, dim=1)
    distances = cross_product_lengths / direction_length
    return distances.cpu()

@ log_time
def mit_tt(ori_pred, head, best_plane, measure):
    point0 , point1 = None, None
    yxf_points = torch.nonzero(torch.from_numpy(ori_pred[::4,::4,::4] == 9)) * 4
    CC_yxf_dis = calculate_points_distance_torch(measure["mitral_cc_real_points"], yxf_points)
    value_min, _ = torch.min(CC_yxf_dis,dim=1)
    if value_min[0] < value_min[1]:
        pointRC, pointLC = measure["mitral_cc_real_points"]  # 第一个值靠近右心房
    else:
        pointLC, pointRC = measure["mitral_cc_real_points"]  # 使用两个点的中间点生成平面，用这两个点矫正平面方向
    measure["pointLC"] = pointLC
    measure["pointRC"] = pointRC
    pointO = torch.mean(measure['hull_point_3d'], dim=0)     # 投影瓣环 中点
    pointsC = upsample_curve(torch.stack((pointLC, pointRC),dim=0), 256)  # 从左到右
    zb_points = torch.nonzero(torch.from_numpy((ori_pred == 6)|(ori_pred == 7)))  # 携带无冠瓣，预防二叶瓣
    yb_points = torch.nonzero(torch.from_numpy((ori_pred == 6)|(ori_pred == 5)))
    # 补充主动脉
    zdm = torch.nonzero(torch.from_numpy(ori_pred == 11))
    zdm_by_dis = calculate_points_distance_torch(
        torch.mean(torch.nonzero(
            torch.from_numpy((ori_pred == 5)|(ori_pred == 6)|(ori_pred == 7))).type(torch.float),
                   dim=0).unsqueeze(0), zdm)
    zdm_point = zdm[zdm_by_dis[0] < 50]
    zb_points = torch.cat((zb_points, zdm_point), dim=0)
    yb_points = torch.cat((yb_points, zdm_point), dim=0)
    for i in range(1,len(pointsC)-1):
        tmp_plane = find_perpendicular_plane_new(torch.stack((pointO.type(torch.float),
                                                          pointsC[i].type(torch.float)),dim=0),
                                             measure["best_plane"].type(torch.float), head["spacing"])
        tmp_plane = check_plane_direction(pointLC.type(torch.float), pointRC.type(torch.float), tmp_plane)
        if point0 is None:
            zb_plane_dis = calculate_points_plane_distance(zb_points, tmp_plane)
            print(i,'zb_plane_dis:',torch.min(zb_plane_dis))
            if torch.min(zb_plane_dis) <= np.sqrt(2) - 1:
                qy_points = measure["mitral_points_1"]
                point1_dis = calculate_points_plane_distance(qy_points, tmp_plane)
                _, min_index = torch.min(torch.abs(point1_dis), dim=0)
                point0 = qy_points[min_index]
        if point0 is not None and point1 is None:
            yb_plane_dis = calculate_points_plane_distance(yb_points, tmp_plane)
            print(i, 'yb_plane_dis:', torch.max(yb_plane_dis))
            if torch.max(yb_plane_dis) <= np.sqrt(2) - 1:
                qy_points = measure["mitral_points_1"]
                point1_dis = calculate_points_plane_distance(qy_points, tmp_plane)
                _, min_index = torch.min(torch.abs(point1_dis), dim=0)
                point1 = qy_points[min_index]
        if point0 is not None and point1 is not None:
            break
    tag_points = torch.stack((point0, point1))
    points_phy = convert_to_physical_coordinates(tag_points.cpu(), head['spacing'])
    measure["mitral_tt_points"] = tag_points
    measure["mitral_tt"] = np.linalg.norm(points_phy[0] - points_phy[-1])

    annulus_plane_dis = calculate_points_plane_distance( measure["mitral_points_2"] , best_plane)
    annulus_plane_dis_tt = calculate_points_plane_distance( measure["mitral_tt_points"] , best_plane)

    tmp_plane_max = copy.deepcopy(best_plane)
    tmp_plane_max[-1] -= torch.abs(torch.max(annulus_plane_dis_tt)/2)
    tmp_plane_min = copy.deepcopy(best_plane)
    tmp_plane_min[-1] += torch.abs(torch.min(annulus_plane_dis)) + 0.5

    annulus_pointsA_proj = project_points_onto_plane_gpu(measure["mitral_point"], tmp_plane_max, True)
    annulus_pointA = torch.mean(annulus_pointsA_proj, dim=0)
    annulus_pointB = project_points_onto_plane_gpu(annulus_pointA.unsqueeze(0), tmp_plane_min)[0]

    measure["mitral_annulus_hight_points"] = torch.stack((annulus_pointA, annulus_pointB), dim=0)
    measure["mitral_annulus_hight_planes"] = torch.stack((tmp_plane_max, tmp_plane_min), dim=0)
    measure["mitral_annulus_hight"] = np.linalg.norm(
        convert_to_physical_coordinates(annulus_pointA.unsqueeze(0).cpu(), head['spacing'])[0] -
        convert_to_physical_coordinates(annulus_pointB.unsqueeze(0).cpu(), head['spacing'])[0])