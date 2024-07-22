import numpy as np
from scipy.interpolate import CubicSpline
import torch
def resamplingA():
    return

def resamplingB(points):
    distances = torch.sqrt(torch.sum((points[1:] - points[:-1]) ** 2, dim=1))
    cumulative_distances = np.cumsum(distances.numpy())
    cumulative_distances = np.insert(cumulative_distances, 0, 0)

    # 创建三次样条插值
    cs_x = CubicSpline(cumulative_distances, points[:, 0].numpy())
    cs_y = CubicSpline(cumulative_distances, points[:, 1].numpy())
    cs_z = CubicSpline(cumulative_distances, points[:, 2].numpy())

    # 生成新的均匀距离点
    n_points = len(points)
    even_distances = np.linspace(0, cumulative_distances[-1], n_points)
    new_x = cs_x(even_distances)
    new_y = cs_y(even_distances)
    new_z = cs_z(even_distances)
    new_points = torch.tensor(np.stack([new_x, new_y, new_z], axis=1), dtype=torch.float64)
    return new_points

def resamplingC(points, distances):
    tmp_dist_0 = max(np.mean(distances)/2, 2)
    tmp_dist = np.percentile(distances, 75)
    tmp_dist_2 = np.percentile(distances, 90)
    tmp_dist_2 = np.percentile(distances, 98) if tmp_dist_2 < tmp_dist*1.5 else tmp_dist_2
    tmp_points = {}
    for i in range(len(distances)):
        if distances[i] > tmp_dist_2:
            if i == len(distances) - 1:
                tmp_points[i] = (points[i] + (points[0] - points[i])/3, points[i] + (points[0] - points[i])*2/3)
            else:
                tmp_points[i] = (points[i] + (points[i+1] - points[i])/3, points[i] + (points[i+1] - points[i])*2/3)
        elif distances[i] > tmp_dist:
            if i==len(distances)-1:
                tmp_points[i] = (points[i] + points[0])/2
            else:
                tmp_points[i] = (points[i] + points[i+1])/2
        elif distances[i] < tmp_dist_0:
            tmp_points[i] = None
    num = 1
    for k,v in tmp_points.items():
        if isinstance(v, torch.Tensor):
            points = torch.cat((points[:k+num], v.unsqueeze(0), points[k+num:]), dim=0)
            num+=1
        elif isinstance(v, tuple):
            points = torch.cat((points[:k + num], torch.stack((v[0], v[1]), dim=0), points[k + num:]), dim=0)
            num+=2
        elif v is None:
            points = torch.cat((points[:k + num], points[k + num +1 :]), dim=0)
            num -= 1
    return points

def curvature_curve(points, model='base'):
    # 三种模型 ， model = 'mixed'  'example'  'base' 如果不为前两种，默认base，
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # 对每个坐标轴进行样条插值
    t = np.arange(len(points))
    interp_x = CubicSpline(t, x)
    interp_y = CubicSpline(t, y)
    interp_z = CubicSpline(t, z)

    def curvature(t):
        dx_dt = interp_x(t, 1)
        d2x_dt2 = interp_x(t, 2)
        dy_dt = interp_y(t, 1)
        d2y_dt2 = interp_y(t, 2)
        dz_dt = interp_z(t, 1)
        d2z_dt2 = interp_z(t, 2)
        # 计算一阶导数的向量
        dr_dt = np.array([dx_dt, dy_dt, dz_dt])
        # 计算二阶导数的向量的范数
        norm_d2r_dt2 = np.linalg.norm([d2x_dt2, d2y_dt2, d2z_dt2])
        # 计算曲率
        curvature = norm_d2r_dt2 / np.linalg.norm(dr_dt) ** 3
        return curvature

    # 曲率
    curvatures = [curvature(t_i) for t_i in t]
    # 距离
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    distances = np.append(distances, np.linalg.norm(points[-1] - points[0]) )

    # 计算每个点的夹角
    extended_points = np.concatenate((points[-1:], points, points[:1]), axis=0)
    angles = [] # 如果出现大于180° ，则出现异常点
    for i in range(1, len(extended_points) - 1):
        # 计算相邻向量
        vector1 = extended_points[i] - extended_points[i - 1]
        vector2 = extended_points[i + 1] - extended_points[i]
        # 计算向量之间的夹角
        angle = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
        angles.append(np.degrees(angle))

    if model == 'mixed':
        # 使用 curvatures + distances + angles 进行 修正
        points = resamplingC(points, distances, angles, curvatures)
    elif model =='example':
        # 使用 distances + angles 进行 修正
        points = resamplingC(points, distances)
        points = resamplingB(points)
    else:
        # 默认的base，只使用 distances 进行 修正
        points = resamplingC(points, distances)

    return points