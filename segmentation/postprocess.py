import numpy
import numpy as np
import torch
from alg_utils.sys import Args
from scipy.ndimage import label
from measure.tool.spendtime import log_time
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree
def process_label(pred, l):
    # 生成二值掩码并计算连通区域，使用 PyTorch Tensor 操作
    tag_mask = (pred == l).to(torch.int)
    # 需要将Tensor转换为numpy，因为scipy不处理Tensor
    tag_mask_numpy = tag_mask.numpy()
    labeled_array, num_features = label(tag_mask_numpy)
    if num_features > 1:
        sizes = np.bincount(labeled_array.ravel())[1:]  # 跳过背景
        largest_component_label = np.argmax(sizes) + 1  # 找到最大组件的标签
        # 返回需要修改的位置，转换回Tensor
        return torch.from_numpy((labeled_array != largest_component_label) & (tag_mask_numpy == 1))
    return None

@ log_time
def postprocess(pred: numpy.ndarray):
    for i in [i for i in range(17,2,-1)]:
        pred[pred == i] = i+1
    data = np.array(pred.clone())
    zdm = (data == 11).astype(int)
    labeled_array, num_features = label(zdm)
    if num_features>1:
        sizes = np.bincount(labeled_array.ravel())
        lst = list(sizes[1:])
        max_index = lst.index(max(lst))
        second_max_index = lst.index(max([x for i, x in enumerate(lst) if i != max_index]))
        tag_1_point = torch.mean(torch.nonzero(torch.from_numpy((labeled_array == max_index + 1))).type(torch.float),dim=0)
        tag_2_point = torch.mean(torch.nonzero(torch.from_numpy((labeled_array == second_max_index + 1))).type(torch.float), dim=0)
        tag_xg_point = torch.mean(torch.nonzero((pred == 5)|(pred == 6)|(pred == 7)).type(torch.float), dim=0)
        if lst[second_max_index] < 5000:
            Tag = max_index + 1
        elif torch.norm(tag_1_point-tag_xg_point) > torch.norm(tag_2_point-tag_xg_point):
            Tag = second_max_index + 1
        else:
            Tag = max_index + 1
        clear_points = torch.nonzero(torch.from_numpy((labeled_array != Tag)&((labeled_array != 0))))
        pred[clear_points[:, 0], clear_points[:, 1], clear_points[:, 2]] = 0

    with ThreadPoolExecutor(max_workers=5) as executor:# 将每个标签的处理提交给线程池
        results = list(executor.map(lambda l: process_label(pred, l), [1,2,9,10,17]))
    for result in results: # 聚合所有需要修改的位置并更新pred
        if result is not None:
            pred[result] = 0
    return pred
