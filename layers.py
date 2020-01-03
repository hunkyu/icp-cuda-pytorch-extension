import torch
import faiss
import numpy as np

import custom_ext as _C


def nn_search(query, ref, ratio=0.5, cur_label=None,
              prev_label=None, gt=False, ignore_label=255):
    """Nearest neighbor search"""
    idx, dist = _C.nn_search(query, ref)
    N = query.size(0)

    # TODO: post-processing time?
    corres = torch.empty(N, 3)
    corres[:, 0] = torch.arange(N)
    corres[:, 1] = idx
    corres[:, 2] = dist

    # skip gt for test (when label is all 0)
    if gt and cur_label.sum() != 0:
        correct_idx = (cur_label == prev_label[corres[:, 1].long()]) \
            & (cur_label != ignore_label)
        corres = corres[correct_idx]
        _, mink = torch.topk(-corres[:, 2], int(N * ratio))
    else:
        _, mink = torch.topk(-corres[:, 2], int(N * ratio))
    return corres[mink, :2].long()


def icp_pytorch(src, dst, max_iter, threshold=0.005, ratio=0.5):
    prev_dist = 0

    for i in range(max_iter):
        # 1. Find Nearest Neighbor
        idx, dist = _C.nn_search(src.cuda(), dst.cuda())
        dst_temp = dst[idx]

        # 2. Compute H matrix
        src_center = src.mean(dim=0)
        dst_temp_center = dst_temp.mean(dim=0)
        src_norm = src - src_center
        dst_temp_norm = dst_temp - dst_temp_center
        h_matrix = torch.mm(src_norm.T, dst_temp_norm)

        # 3. SVD
        U, S, V = torch.svd(h_matrix)  # FIXME: very slow

        # 4. Rotation matrix and translation vector
        R = torch.mm(U, V.T)
        t = dst_temp_center - torch.mm(R, src_center.unsqueeze(1)).squeeze()

        # 5. Transform
        src = torch.mm(src, R) + t.unsqueeze(0)
        mean_dist = dist.mean()
        if torch.abs(mean_dist - prev_dist) < threshold:
            break
        prev_dist = mean_dist

    _, mink = torch.topk(-dist, int(src.size(0) * ratio))
    corres = torch.empty(src.size(0), 2)
    corres[:, 0] = torch.arange(src.size(0))
    corres[:, 1] = idx

    return corres[mink].long()

