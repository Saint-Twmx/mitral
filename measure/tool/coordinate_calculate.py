# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:21:08 2024

@author: admin
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

def convert_to_physical_coordinates(points, spacing):
    physical_points = []
    for point in points:
        physical_point = [point[i] * spacing[i] for i in range(len(point))]
        physical_points.append(physical_point)
    return np.array(physical_points)

def convert_to_physical_coordinates_gpu(points, spacing):
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    if not isinstance(spacing, torch.Tensor):
        spacing = torch.tensor(spacing, dtype=torch.float32)
    spacing = spacing.to(points.device)
    spacing = spacing.unsqueeze(0)
    physical_points = points * spacing
    return physical_points

def rotate_points_to_align_with_x(points, normal):
    axis_of_rotation = np.cross(normal, [0, 0, 1]) # XY 平面 Z 为1    ZY平面  X为1
    # Angle of rotation is the angle between the normal vector and Z axis
    angle_of_rotation = np.arccos(np.dot(normal, [0, 0, 1]) / np.linalg.norm(normal))
    # Create a rotation object
    rotation = R.from_rotvec(axis_of_rotation * angle_of_rotation)
    rotated_points = rotation.apply(points)
    return rotated_points

def project_to_zy_plane(points):
    return points[:, :2]

def calculate_perimeter(physical_points, types=2, line=False):
    # 计算点集之间距离和，近似为周长，
    n = len(physical_points)
    perimeter = 0.0
    if line:
        differences = physical_points[1:] - physical_points[:-1]
        distances = np.linalg.norm(differences, axis=1)
        perimeter = np.sum(distances)
    else:
        for i in range(n):
            if types == 2:
                point1 = physical_points[i][:2]
                point2 = physical_points[(i + 1) % n][:2] # To ensure closure with the first point
            else:
                point1 = physical_points[i]
                point2 = physical_points[(i + 1) % n]
            perimeter += np.linalg.norm(point2 - point1)
    return perimeter

def calculate_area_of_polygon_2d(points):
    """
    多边形面积公式，二维点集近似计算面积，默认截取前两个轴
    """
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i][:2]
        x2, y2 = points[(i + 1) % n][:2]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def grid_based_downsampling(points, grid_size):
    # 确定点集的边界
    min_point = torch.min(points, dim=0)[0]
    max_point = torch.max(points, dim=0)[0]
    # 计算网格的尺寸
    dims = ((max_point - min_point) / grid_size).int() + 1
    # 将点映射到网格中
    grid = {}
    for point in points:
        grid_idx = tuple(((point - min_point) / grid_size).int().tolist())
        if grid_idx not in grid:
            grid[grid_idx] = []
        grid[grid_idx].append(point)
    downsampled_points = []
    for idx in grid:
        downsampled_points.append(grid[idx][0])

    return torch.stack(downsampled_points)


def calculate_area_of_3d_polygon_with_centroid(points):
    """
    给一组点，默认够长凸多边形，先计算一个中心线，然后利用三角形面积，累加成总面积
    """
    if len(points) < 3:
        return 0
    points_array = np.array(points)
    centroid = points_array.mean(axis=0)
    total_area = 0
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        vec1 = p1 - centroid
        vec2 = p2 - centroid
        triangle_area = np.linalg.norm(np.cross(vec1, vec2)) / 2
        total_area += triangle_area
    return total_area

def determine_plane(points: torch.Tensor) : # 点集，最小二乘法，生成一个plane
    x = points.clone()
    x_T = x.T.double()
    x_TX = torch.mm(x_T, x.double())
    x_TX_1 = torch.inverse(x_TX).double()
    plane = (
        torch.cat([x_TX_1.mm(x_T).sum(1), torch.tensor([-1]).to(x.device)], dim=-1)
        * 1000
    )
    return plane.float()


def determine_plane(points: torch.Tensor,):
    x = points.clone()
    x_T = x.T.double()
    x_TX = torch.mm(x_T, x.double())
    x_TX_1 = torch.inverse(x_TX).double()
    plane = (
        torch.cat([x_TX_1.mm(x_T).sum(1), torch.tensor([-1]).to(x.device)], dim=-1)
        * 1000
    )
    plane = plane.float()
    return plane


def fit_plane_pca(points: torch.Tensor):
    centroid = points.mean(dim=0)
    centered_points = points - centroid
    H = torch.mm(centered_points.t(), centered_points)
    eigenvalues, eigenvectors = torch.linalg.eigh(H)
    normal_vector = eigenvectors[:, 0]
    d = -torch.dot(normal_vector, centroid)
    plane_parameters = torch.cat([normal_vector, d.unsqueeze(0)])
    return plane_parameters