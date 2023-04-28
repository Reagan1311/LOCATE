import torch.nn as nn
import torch
import numpy as np

__all__ = ['normalize_minmax', 'concentration_loss']


def normalize_minmax(cams, eps=1e-15):
    B, _, _ = cams.shape
    min_value, _ = cams.view(B, -1).min(1)
    cams_minmax = cams - min_value.view(B, 1, 1)
    max_value, _ = cams_minmax.view(B, -1).max(1)
    cams_minmax /= max_value.view(B, 1, 1) + eps
    return cams_minmax


def get_variance(part_map, x_c, y_c):
    h, w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h, w)

    v_x_map = (x_map - x_c) * (x_map - x_c)
    v_y_map = (y_map - y_c) * (y_map - y_c)

    v_x = (part_map * v_x_map).sum()
    v_y = (part_map * v_y_map).sum()
    return v_x, v_y


def get_coordinate_tensors(x_max, y_max):
    x_map = np.tile(np.arange(x_max), (y_max, 1)) / x_max * 2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max, 1)).T / y_max * 2 - 1.0

    x_map_tensor = torch.from_numpy(x_map.astype(np.float32)).cuda()
    y_map_tensor = torch.from_numpy(y_map.astype(np.float32)).cuda()

    return x_map_tensor, y_map_tensor


def get_center(part_map, self_referenced=False):
    h, w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h, w)

    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    if self_referenced:
        x_c_value = float(x_center.cpu().detach())
        y_c_value = float(y_center.cpu().detach())
        x_center = (part_map * (x_map - x_c_value)).sum() + x_c_value
        y_center = (part_map * (y_map - y_c_value)).sum() + y_c_value

    return x_center, y_center


def get_centers(part_maps, detach_k=True, epsilon=1e-3, self_ref_coord=False):
    H, W = part_maps.shape
    part_map = part_maps + epsilon
    k = part_map.sum()
    part_map_pdf = part_map / k
    x_c, y_c = get_center(part_map_pdf, self_ref_coord)
    centers = torch.stack((x_c, y_c), dim=0)
    return centers


def batch_get_centers(pred_norm):
    B, H, W = pred_norm.shape

    centers_list = []
    for b in range(B):
        centers_list.append(get_centers(pred_norm[b]).unsqueeze(0))
    return torch.cat(centers_list, dim=0)


# Code borrowed from SCOPS https://github.com/NVlabs/SCOPS
def concentration_loss(pred):
    # b x h x w
    B, H, W = pred.shape
    tmp_max, tmp_min = pred.max(-1)[0].max(-1)[0].view(B, 1, 1), \
                       pred.min(-1)[0].min(-1)[0].view(B, 1, 1)

    pred_norm = ((pred - tmp_min) / (tmp_max - tmp_min + 1e-10))  # b x 28 x 28

    loss = 0
    epsilon = 1e-3
    centers_all = batch_get_centers(pred_norm)
    for b in range(B):
        centers = centers_all[b]
        # normalize part map as spatial pdf
        part_map = pred_norm[b, :, :] + epsilon  # prevent gradient explosion
        k = part_map.sum()
        part_map_pdf = part_map / k
        x_c, y_c = centers
        v_x, v_y = get_variance(part_map_pdf, x_c, y_c)
        loss_per_part = (v_x + v_y)
        loss = loss_per_part + loss
    loss = loss / B
    return loss
