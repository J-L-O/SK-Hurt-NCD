from typing import Tuple

import torch
import numpy as np


@torch.no_grad()
def compute_transfer_flow(
        features: np.ndarray,
        labels: np.ndarray,
        kernel: str,
        num_classes: int,
        device: torch.device,
        return_all=False) -> torch.Tensor:
    num_unlabeled_samples = len(labels)

    mmd_matrix_list = []
    weight_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=device)

    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue

            indices_i = labels == i
            indices_j = labels == j

            features_i = torch.tensor(features[indices_i], device=device)
            features_j = torch.tensor(features[indices_j], device=device)

            num_samples_i = features_i.shape[0]
            num_samples_j = features_j.shape[0]

            if num_samples_i == 0 or num_samples_j == 0:
                continue

            mmd = mmd_cal(features_i, features_j, kernel, return_all=return_all)

            weight = num_samples_i * num_samples_j / (num_unlabeled_samples * (num_unlabeled_samples - 1))
            weight_matrix[i, j] = weight

            if len(mmd_matrix_list) == 0:
                mmd_matrix_list = [torch.zeros((num_classes, num_classes), dtype=torch.float32, device=device)
                                   for _ in range(len(mmd))]

            for k in range(len(mmd)):
                mmd_matrix_list[k][i, j] = mmd[k]

    transfer_flow = (torch.stack(mmd_matrix_list) * weight_matrix.unsqueeze(0)).sum(dim=(1, 2))

    return transfer_flow


def bootstrap_transfer_flow(
        features: np.ndarray,
        labels: np.ndarray,
        num_samples: int,
        kernel: str,
        num_classes: int,
        device: torch.device,
        return_all=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_flows = []

    for i in range(num_samples):
        sample_indices = np.random.choice(len(features), len(features), replace=True)
        sample_features = features[sample_indices]
        sample_labels = labels[sample_indices]

        sample_flow = compute_transfer_flow(sample_features, sample_labels, kernel, num_classes,
                                               device, return_all=return_all)
        sample_flows.append(sample_flow)

    sample_flows = torch.stack(sample_flows)
    std, mean = torch.std_mean(sample_flows, dim=0, unbiased=True)

    standard_error = std / torch.sqrt(torch.tensor(num_samples, dtype=torch.float, device=device))

    return mean, std, standard_error


def squared_euclidean_cdist(x, y):
    return torch.cdist(x, y, p=2)**2


def l1_cdist(x, y):
    return torch.cdist(x, y, p=1)


def chunked_cdist(total, dist_fn, max_size=1000):
    if total.shape[0] < max_size:
        return dist_fn(total, total)
    else:
        n = total.shape[0]
        n_tiles = int(np.ceil(n / max_size))
        tiles = torch.split(total, max_size, dim=0)
        distances = torch.zeros((n, n), device=total.device)
        for i in range(n_tiles):
            distances_tile = []
            for j in range(n_tiles):
                tiles_dist = dist_fn(tiles[i], tiles[j])
                distances_tile.append(tiles_dist)

            distances_tile = torch.cat(distances_tile, dim=1)
            distances[i * max_size:(i + 1) * max_size] = distances_tile

        return distances


def chunked_exp_sum(distances, bandwidth_list, max_size=1000):
    if distances.shape[0] < max_size:
        kernel_val = [torch.exp(-distances / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return kernel_val
    else:
        # Save GPU memory
        device = distances.device
        distances = distances.cpu()

        n = distances.shape[0]
        n_tiles = int(np.ceil(n / max_size))
        tiles = torch.split(distances, max_size, dim=0)
        kernel_val = [torch.zeros_like(distances, device=device) for _ in range(len(bandwidth_list))]

        for i in range(n_tiles):
            tile = tiles[i].to(device)

            for j, bandwidth_temp in enumerate(bandwidth_list):
                exp = torch.exp(-tile / bandwidth_temp)
                kernel_val[j][i * max_size:(i + 1) * max_size] = exp

        return kernel_val


def kernel_fn(source, target, dist_fn, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = source.shape[0] + target.shape[0]
    total = torch.cat([source, target], dim=0)

    l2_distances = chunked_cdist(total, dist_fn)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distances.detach()) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = chunked_exp_sum(l2_distances, bandwidth_list)

    return kernel_val


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    return kernel_fn(source, target, squared_euclidean_cdist, kernel_mul, kernel_num, fix_sigma)


def laplacian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    return kernel_fn(source, target, l1_cdist, kernel_mul, kernel_num, fix_sigma)


def get_kernel_fn(kernel_type):
    if kernel_type == 'gaussian':
        return gaussian_kernel
    elif kernel_type == 'laplacian':
        return laplacian_kernel
    else:
        raise ValueError(f'Unknown kernel type: {kernel_type}')

def prob_map(sp, tp):

    total = torch.cat([sp, tp], dim=0).unsqueeze(1)
    total1 = torch.cat([sp, tp], dim=0).unsqueeze(0)
    prob = torch.mm(total, total1)
    return prob


def mmd_from_kernels(kernels, n, m):
    XX = kernels[:n, :n]
    YY = kernels[-m:, -m:]

    XY = kernels[:n, -m:]
    YX = kernels[-m:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

    mmd = (XX + XY).sum() + (YX + YY).sum()
    return mmd


def mmd_cal(source, target, kernel="gaussian", kernel_mul=2.0, kernel_num=5, fix_sigma=None, return_all=False):
    kernel_fn = get_kernel_fn(kernel)

    n = source.shape[0]
    m = target.shape[0]
    kernels = kernel_fn(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    if return_all:
        mmds = [mmd_from_kernels(kernels_bandwidth, n, m) for kernels_bandwidth in kernels]
    else:
        kernel_sum = sum(kernels)

        mmds = [mmd_from_kernels(kernel_sum, n, m)]

    return mmds


if __name__ == "__main__":
    # 样本数量可以不同，特征数目必须相同

    # 100和90是样本数量，50是特征数目
    data_1 = torch.tensor(np.random.normal(loc=0, scale=10, size=(100, 50)))
    data_2 = torch.tensor(np.random.normal(loc=0, scale=9, size=(80, 50)))

    print("MMD Loss:", mmd_cal(data_1, data_2))

