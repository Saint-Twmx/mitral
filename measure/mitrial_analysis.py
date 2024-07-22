import numpy as np
import torch
from scipy.spatial import cKDTree
from functools import reduce
import operator
from measure.mitral_bestplane import project_points_onto_plane_gpu
from measure.tool.concaveHull import get_outer_contour_3d
from measure.tool.resample_3d import resample_curve
from measure.tool.coordinate_calculate import (
    convert_to_physical_coordinates_gpu,
    fit_plane_pca
)
from scipy.ndimage import label
from measure.mitral_bestplane import calculate_points_plane_distance
from measure.mitral_annulus import calculate_points_distance_torch

def point_to_line_distance(points, point):
    assert points.shape == (2, 3), "点集应该是两个三维点"
    assert point.shape == (3,), "点应该是一个三维向量"
    A, B = points[0], points[1]
    AB = B - A
    AP = point - A
    cross_product = torch.cross(AP, AB)
    cross_norm = torch.norm(cross_product)
    AB_norm = torch.norm(AB)
    distance = cross_norm / AB_norm
    return distance.item()


def plane_angle(plane1, plane2, neg=False):
    normal_septum = plane1[:3]
    if neg:
        normal_septum = -plane1[:3]
    normal_best = plane2[:3]
    length_septum = torch.norm(normal_septum)
    length_best = torch.norm(normal_best)
    dot_product = torch.dot(normal_septum, normal_best)
    cos_angle = dot_product / (length_septum * length_best)
    angle_degrees = torch.rad2deg(torch.acos(cos_angle))
    return angle_degrees

def mit_int_septum(measure, ori_pred):
    # 房间隔夹角
    best_plane = measure['best_plane']

    pointA = torch.nonzero(torch.from_numpy(ori_pred[::2,::2,::2] == 9)) * 2
    pointB = torch.nonzero(torch.from_numpy(ori_pred[::2,::2,::2] == 10)) * 2
    treeB = cKDTree(pointA.numpy())
    distances, indices = treeB.query(pointB.numpy(), distance_upper_bound=np.sqrt(5))
    adjacent_coords = torch.from_numpy(pointB.numpy()[distances <= np.sqrt(4)]).type(torch.float)

    septum_plane = fit_plane_pca(adjacent_coords)

    projection_point_proj = project_points_onto_plane_gpu(adjacent_coords, septum_plane, True)
    hull_point_proj, _ = get_outer_contour_3d(np.array(projection_point_proj.cpu()), np.array(best_plane), types=True)
    three_points = resample_curve(hull_point_proj, 3)
    center_point = torch.mean(adjacent_coords, dim=0)
    three_points = (center_point + three_points) / 2

    yxs_point = torch.mean(pointA.type(torch.float), dim=0)
    distance = (septum_plane[:3].dot(yxs_point) + septum_plane[3]) / septum_plane[:3].norm()

    if distance < 0:
        angle = plane_angle(septum_plane, best_plane, neg=True)
    else:
        angle = plane_angle(septum_plane, best_plane)  #best_plane方向往左心室，所以要调整 septum_plane 方向为右心房。

    measure['septum_plane'] = septum_plane
    measure["interatrial_septum_three_points"] = three_points
    measure["interatrial_septum_annluls_angle"] = float(angle)

def mit_non_planarity(measure):
    # 非平面化角度
    center = (measure["mitral_cc_real_points"][0] + measure["mitral_cc_real_points"][1])/2
    CA, CB = measure["mitral_ap_points"] - center
    cos_theta = torch.dot(CA, CB) / (torch.norm(CA) * torch.norm(CB))
    theta_deg = torch.rad2deg(torch.arccos(cos_theta))
    measure["non_planarity"] = float(theta_deg)

def mitral_papillary_muscle_analysis(ori_pred, measure, voxel_volume, head):
    rtj_voxels = len(torch.nonzero(torch.from_numpy(ori_pred==4)))
    rtj_volume = rtj_voxels * voxel_volume
    measure["papillary_muscle_volume"] = float(rtj_volume/1000) # 乳头肌 体积
    # 前侧乳头肌  后侧乳头肌 距离瓣环平面 距离
    labeled_mask, num_features = label(ori_pred==4)
    sizes = np.bincount(labeled_mask.ravel())
    other_indices = np.delete(np.arange(len(sizes)), np.argmax(sizes))
    split_plane = fit_plane_pca(torch.nonzero(torch.from_numpy(ori_pred==18)).type(torch.float))
    tmp_dict = []
    for i in other_indices:
        tmp_points = torch.nonzero(torch.from_numpy(labeled_mask==i))
        tmp_split_plane_dis = calculate_points_plane_distance(tmp_points, split_plane)
        tmp_dict.append((i, torch.abs(torch.min(tmp_split_plane_dis)), len(tmp_points)))
    split_num = np.mean([t[1] for t in tmp_dict])
    anterior = [t for t in tmp_dict if t[1]<=split_num]
    posterior = [t for t in tmp_dict if t[1]>split_num]
    anterior_index = [t for t in anterior if t[2]==max([t[2] for t in anterior])][0][0]
    posterior_index = [t for t in posterior if t[2] == max([t[2] for t in posterior])][0][0]
    anterior_points = torch.nonzero(torch.from_numpy(labeled_mask == anterior_index))
    posterior_points = torch.nonzero(torch.from_numpy(labeled_mask == posterior_index))
    anterior_plane_dis = calculate_points_plane_distance(anterior_points, measure["best_plane"])
    posterior_plane_dis = calculate_points_plane_distance(posterior_points, measure["best_plane"])
    anterior_point = anterior_points[torch.max(anterior_plane_dis,dim=0)[1]]
    posterior_point = posterior_points[torch.max(posterior_plane_dis, dim=0)[1]]
    anterior_point_proj = project_points_onto_plane_gpu(anterior_point.unsqueeze(0), measure["best_plane"]).cpu()
    posterior_point_proj = project_points_onto_plane_gpu(posterior_point.unsqueeze(0), measure["best_plane"]).cpu()
    points_phy = convert_to_physical_coordinates_gpu(torch.stack((anterior_point, anterior_point_proj[0]), dim=0),
                                                     head['spacing'])
    measure["anterior_papillary_muscle_to_mitral_valve_annular_distance"] = float(np.linalg.norm(points_phy[0] - points_phy[-1]))
    points_phy = convert_to_physical_coordinates_gpu(torch.stack((posterior_point, posterior_point_proj[0]), dim=0),
                                                     head['spacing'])
    measure["posterior_papillary_muscle_to_mitral_valve_annular_distance"] = float(np.linalg.norm(points_phy[0] - points_phy[-1]))


def numerical_calculation(measure, ori_pred, head):
    voxel_volume = reduce(operator.mul, head['spacing']) # 立方毫米

    zxs_points = torch.nonzero(torch.from_numpy(ori_pred==16))
    zxs_voxels = len(zxs_points)
    zxs_volume = zxs_voxels * voxel_volume
    measure["left_ventricular_volume"] = zxs_volume/1000  #左心室体积 容积

    zxs_plane_dis = calculate_points_plane_distance(zxs_points.cuda(), measure["best_plane"].cuda())
    _, max_index = torch.max(torch.abs(zxs_plane_dis),dim=0)
    points_phy = convert_to_physical_coordinates_gpu(
        torch.stack((zxs_points[max_index], project_points_onto_plane_gpu(
            zxs_points[max_index].unsqueeze(0), measure["best_plane"]).cpu()[0]),
                    dim=0), head['spacing'])  # 左室心尖距离瓣环平面距离
    measure["left_ventricle_apex_to_mitral_valve_annular_distance"] = np.linalg.norm(points_phy[0] - points_phy[-1])

    zxj_voxels = len(torch.nonzero(torch.from_numpy(ori_pred==17)))
    zxj_volume = zxj_voxels * voxel_volume
    measure["left_ventricular_myocardial_volume"] = zxj_volume/1000 # 左心肌 体积

    mitral_papillary_muscle_analysis(ori_pred, measure, voxel_volume, head)  # 乳头肌分析

    mit_int_septum(measure, ori_pred)  #房间隔 与 瓣环平面 夹角

    mit_non_planarity(measure)      # 非平面化角度

    A2 = measure["A2_points_curve_dis"]
    P2 = measure["P2_points_curve_dis"]
    AP = measure["mitral_ap"]
    CC = measure["mitral_cc_real"]
    AH = measure["mitral_annulus_hight"]
    measure["leaflet_to_annulus_ratio"] = (A2 + P2) / AP# 瓣叶 - 瓣环指数 =（A2+P2）/瓣环前后径
    measure["coaptation_index"] = (A2 + P2 - AP) / 2 # 对合指数=（前叶+后叶- 瓣环前后径）/2
    measure["AHCWR"] = AH / CC # 二尖瓣环高度与连合宽度比（AHCWR）

    #主动脉瓣环 - 二尖瓣瓣环  夹角
    zdm = torch.nonzero(torch.from_numpy(ori_pred == 11))
    zdm_by_dis = calculate_points_distance_torch(
        torch.mean(torch.nonzero(
            torch.from_numpy((ori_pred == 5)|(ori_pred == 6)|(ori_pred == 7))).type(torch.float),
                   dim=0).unsqueeze(0), zdm)
    zdm_point = zdm[zdm_by_dis[0] < 55]
    zxs_cen_point = torch.mean(zxs_points.type(torch.float),dim=0)
    zdm_point_plane = fit_plane_pca(zdm_point.type(torch.float))
    zxs_zdmplane_dis = calculate_points_plane_distance(zxs_cen_point.unsqueeze(0), zdm_point_plane)
    if zxs_zdmplane_dis[0] > 0:
        zdm_point_plane = torch.cat([-zdm_point_plane[:3], zdm_point_plane[3].unsqueeze(0)])
    step = 1
    while step < 100:
        zdm_point_plane[-1] += 5
        zdm_plane_dis = calculate_points_plane_distance(zdm_point, zdm_point_plane)
        if torch.sum(zdm_plane_dis < 0).item() / len(zdm_point) < 0.01:
            break
        step+=1

    measure["aortic_valve_annulus_and_mitral_valve_annulus_angle"] = float(plane_angle(zdm_point_plane, measure["best_plane"]))

    return