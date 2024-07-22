import torch
import numpy as np
from measure.mitral_annulus import calculate_points_distance_torch
from measure.tool.coordinate_calculate import convert_to_physical_coordinates, convert_to_physical_coordinates_gpu
import copy
from measure.tool.spendtime import log_time

def my_array_split(mitral_hull_point, mitral_hull_point_index, split=3, spacing=None):
    # 依据距离，切分索引
    check_points = mitral_hull_point[mitral_hull_point_index]
    if spacing:
        check_points = convert_to_physical_coordinates_gpu(check_points, spacing)
    diff = check_points[1:] - check_points[:-1]
    distances = torch.norm(diff, dim=1)
    # 计算所有相邻点之间距离的总和
    total_distance = torch.sum(distances)
    if split == 0:
        split_distance = total_distance / 2
        tmp_sum = 0
        for i in range(1,len(check_points)):
            tmp_sum += torch.norm(check_points[i] - check_points[i - 1])
            if tmp_sum >= split_distance:
                if tmp_sum-split_distance<torch.norm(check_points[i] - check_points[i - 1]):
                    return mitral_hull_point_index[i]
                else:
                    return mitral_hull_point_index[i-1]
    else:
        split_distance = total_distance/split
        index_triplet = []
        tmp = [mitral_hull_point_index[0]]
        tmp_sum = 0
        for i in range(1,len(check_points)):
            if tmp_sum <= split_distance:
                tmp.append(mitral_hull_point_index[i] )
                tmp_sum += torch.norm(check_points[i] - check_points[i-1])
            else:
                index_triplet.append(np.array(tmp))
                tmp = [mitral_hull_point_index[i]]
                tmp_sum = 0
            if i+1 == len(check_points):
                index_triplet.append(np.array(tmp))
        return index_triplet

def auto_pair(listA, listB, expand = 1):
    if len(listA) < len(listB):
        min_list = listA
        max_list = listB
        seq = "AB"
    else:
        min_list = listB
        max_list = listA
        seq = "BA"

    l1 = len(min_list)
    l2 = len(max_list)

    A = l2 // l1  # 每个元祖最小配对只
    B = l2 % l1  # 最小list中有多少个配对 A+1个
    C = []
    tmp = 0
    for i in range(l1):
        if l1 - (i + 1) >= B:
            for j in range(A):
                C.append(tuple([min_list[i], max_list[tmp]]))
                tmp += 1
        else:
            for j in range(A + 1):
                C.append(tuple([min_list[i], max_list[tmp]]))
                tmp += 1
        if expand > 0:
            for j in range(expand):
                if tmp + j < len(max_list) - 1:
                    C.append(tuple([min_list[i], max_list[tmp + j]]))
    return C,seq

def calculate_angle(pointsA, pointsB):
    if pointsA.shape != (2, 3) or pointsB.shape != (2, 3):
        raise ValueError("Each points tensor must be of shape [2, 3]")
    vectorA = pointsA[1] - pointsA[0]
    vectorB = pointsB[1] - pointsB[0]
    dot_product = torch.dot(vectorA, vectorB)
    normA = torch.norm(vectorA)
    normB = torch.norm(vectorB)
    cos_theta = dot_product / (normA * normB)
    theta = torch.acos(cos_theta) * (180 / torch.pi)
    return theta


@log_time
def mit_cc_ap(ori_pred, head, best_plane, measure):
    data = ori_pred.copy()

    ########## cc_real
    hull_point_3d = measure["hull_point_3d_resample"]
    mitral_hull_point = measure["mitral_hull_point_resample"]

    mitral_point_1 = torch.nonzero(torch.from_numpy(data == 1)) # 前叶
    mitral_point_2 = torch.nonzero(torch.from_numpy(data == 2)) # 后叶

    hull_dis_1 = calculate_points_distance_torch(mitral_hull_point, mitral_point_1)
    hull_dis_2 = calculate_points_distance_torch(mitral_hull_point, mitral_point_2)
    mitral_hull_point_1_index = []
    mitral_hull_point_2_index = []
    for l in range(len(mitral_hull_point)):
        if torch.min(hull_dis_1[l]) < torch.min(hull_dis_2[l]):
            mitral_hull_point_1_index.append(l)
        else:
            mitral_hull_point_2_index.append(l)

    if all((mitral_hull_point_1_index[i] == mitral_hull_point_1_index[i - 1] + 1) for i in range(1, len(mitral_hull_point_1_index))):
        # 调整 mitral_hull_point_2_index 为连续索引
        lst = copy.deepcopy(mitral_hull_point_2_index)
        break_points = [i for i in range(1, len(lst)) if lst[i] - lst[i-1] > 1]
        if break_points:# 不为空的时候，才需要调整
            mitral_hull_point_2_index = lst[break_points[0]:] + lst[:break_points[0]]
    else:
        # 调整 mitral_hull_point_1_index 为连续索引
        lst = copy.deepcopy(mitral_hull_point_1_index)
        break_points = [i for i in range(1, len(lst)) if lst[i] - lst[i - 1] > 1]
        if break_points:  # 不为空的时候，才需要调整
            mitral_hull_point_1_index = lst[break_points[0]:] + lst[:break_points[0]]

    mitral_hull_point_1_index = mitral_hull_point_1_index[1:-1]
    mitral_hull_point_2_index = mitral_hull_point_2_index[1:-1] # 交接点有概率错位，索性丢了

    # TT 使用 前叶
    measure["mitral_points_1"] = mitral_hull_point[mitral_hull_point_1_index]

    cc_r_h_pointA = (hull_point_3d[mitral_hull_point_1_index[0]] + hull_point_3d[mitral_hull_point_2_index[-1]])/2
    cc_r_h_pointB = (hull_point_3d[mitral_hull_point_1_index[-1]] + hull_point_3d[mitral_hull_point_2_index[0]])/2
    cc_r_m_pointA = (mitral_hull_point[mitral_hull_point_1_index[0]] + mitral_hull_point[mitral_hull_point_2_index[-1]])/2
    cc_r_m_pointB = (mitral_hull_point[mitral_hull_point_1_index[-1]] + mitral_hull_point[mitral_hull_point_2_index[0]])/2
    mitral_cc_real_proj = np.linalg.norm(
        convert_to_physical_coordinates(cc_r_h_pointA.unsqueeze(0), head['spacing'])[0] -
        convert_to_physical_coordinates(cc_r_h_pointB.unsqueeze(0), head['spacing'])[0])
    mitral_cc_real = np.linalg.norm(
        convert_to_physical_coordinates(cc_r_m_pointA.unsqueeze(0), head['spacing'])[0] -
        convert_to_physical_coordinates(cc_r_m_pointB.unsqueeze(0), head['spacing'])[0])
    measure["mitral_cc_real_proj"] = mitral_cc_real_proj
    measure["mitral_cc_real_proj_points"] = torch.stack((cc_r_h_pointA, cc_r_h_pointB), dim=0)
    measure["mitral_cc_real"] = mitral_cc_real
    measure["mitral_cc_real_points"] = torch.stack((cc_r_m_pointA, cc_r_m_pointB), dim=0)

    ############ cc
    mitral_hull_point_2_index_triplet = my_array_split(mitral_hull_point, mitral_hull_point_2_index, split=3)
    pair, _ = auto_pair(mitral_hull_point_2_index_triplet[0], mitral_hull_point_2_index_triplet[-1][::-1], expand=2)
    cc_proj_dis = 0
    cc_dis = 0
    for ij in pair:
        points = hull_point_3d[list(ij)]
        tmp_cc_proj_dis = torch.dist(points[0], points[1]).item()
        if tmp_cc_proj_dis > cc_proj_dis:
            cc_proj_dis_points = points
            cc_proj_dis = tmp_cc_proj_dis
        points = mitral_hull_point[list(ij)]
        tmp_cc_dis = torch.dist(points[0], points[1]).item()
        if tmp_cc_dis > cc_dis:
            cc_dis_points = points
            cc_dis = tmp_cc_dis

    cc_proj_ph_points = convert_to_physical_coordinates(cc_proj_dis_points, head['spacing'])
    mitral_cc_proj = np.linalg.norm(cc_proj_ph_points[0] - cc_proj_ph_points[1])

    cc_ph_points = convert_to_physical_coordinates(cc_dis_points, head['spacing'])
    mitral_cc = np.linalg.norm(cc_ph_points[0] - cc_ph_points[1])

    measure["mitral_cc_proj"] = mitral_cc_proj
    measure["mitral_cc_proj_points"] = cc_proj_dis_points

    measure["mitral_cc"] = mitral_cc
    measure["mitral_cc_points"] = cc_dis_points


    ##############ap  大胆点  尝试物理 中值
    maxindex2 = my_array_split(mitral_hull_point, mitral_hull_point_1_index, split=0, spacing=head['spacing'])
    maxindex1 = my_array_split(mitral_hull_point, mitral_hull_point_2_index, split=0, spacing=head['spacing'])
    #     #20210418调整：判定“相切”角度，修正角度
    ap_m_pointA = mitral_hull_point[maxindex1]
    ap_m_pointB = mitral_hull_point[maxindex2]

    angle = calculate_angle(torch.stack((ap_m_pointA, ap_m_pointB), dim=0),cc_dis_points)
    if angle < 80:
        print("warning!!!!AP/CC夹角异常")
    print(f"AP/CC夹角:{angle}")

    mitral_ap_proj = np.linalg.norm(
        convert_to_physical_coordinates(hull_point_3d[maxindex2].unsqueeze(0), head['spacing'])[0] -
        convert_to_physical_coordinates(hull_point_3d[maxindex1].unsqueeze(0), head['spacing'])[0])
    mitral_ap = np.linalg.norm(
        convert_to_physical_coordinates(mitral_hull_point[maxindex2].unsqueeze(0), head['spacing'])[0] -
        convert_to_physical_coordinates(mitral_hull_point[maxindex1].unsqueeze(0), head['spacing'])[0])

    measure["mitral_ap_proj"] = mitral_ap_proj
    measure["mitral_ap_proj_points"] = torch.stack((hull_point_3d[maxindex2], hull_point_3d[maxindex1]), dim=0)

    measure["mitral_ap"] = mitral_ap
    measure["mitral_ap_points"] = torch.stack((mitral_hull_point[maxindex2], mitral_hull_point[maxindex1]), dim=0)

    print(f" mitral_ap_points ：{measure['mitral_ap_points']}")
    print(f" maxindex2 ：{maxindex2}")
    print(f" maxindex1 ：{maxindex1}")
    print(f" mitral_hull_point_1_index ：{mitral_hull_point_1_index}")
    print(f" mitral_hull_point_2_index ：{mitral_hull_point_2_index}")

    # 只用后叶
    # 20240528 新增 瓣环高度
    measure["mitral_points_2"] = mitral_hull_point[mitral_hull_point_2_index]