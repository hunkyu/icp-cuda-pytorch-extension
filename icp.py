import torch
import faiss
import numpy as np

import custom_ext as _C


def icp(src, dst, max_iter, threshold=0.005, ratio=0.5):
    prev_err = 0

    for i in range(max_iter):
        # 1. Find Nearest Neighbor
        idx = _C.nn_search(src, dst).long()
        dst_temp = dst[idx]

        # 2. Compute H matrix
        src_center = src.mean(dim=0)
        dst_temp_center = dst_temp.mean(dim=0)
        src_norm = src - src_center
        dst_temp_norm = dst_temp - dst_temp_center
        h_matrix = torch.mm(src_norm.T, dst_temp_norm)

        # 3. SVD
        U, S, V = torch.svd(h_matrix)

        # 4. Rotation matrix and translation vector
        R = torch.mm(U, V)
        t = dst_temp_center - torch.mm(R, src_center.unsqueeze(1)).squeeze()

        # 5. Transform
        src = torch.mm(src, R) + t.unsqueeze(0)
        err = torch.sqrt(torch.norm(src - dst_temp, dim=1))
        mean_err = err.mean()
        prev_err = mean_err
        if torch.abs(mean_err - prev_err) < threshold:
            break

    _, mink = torch.topk(-err, int(src.size(0) * ratio))
    corres = torch.empty(src.size(0), 2)
    corres[:, 0] = torch.arange(src.size(0))
    corres[:, 1] = idx

    return corres[mink].long()


def icp_faiss(src, dst, d=3, ratio=0.5):
    res = faiss.StandardGpuResources()  # TODO: global faiss gpu res
    index = faiss.GpuIndexFlat(res, d, faiss.METRIC_L2)
    if isinstance(src, np.ndarray):
        index.add(np.ascontiguousarray(dst))
        D, I = index.search(np.ascontiguousarray(src), 1)
        corres = np.concatenate(
            (np.expand_dims(np.arange(I.shape[0]), 1), I), 1)
        topk = D.argsort(axis=0)[:int(D.shape[0] * ratio)].squeeze()
        return corres[topk]
    else:
        # FIXME
        def swig_ptr_from_FloatTensor(x):
            return faiss.cast_integer_to_float_ptr(
                x.storage().data_ptr() + x.storage_offset() * 4)

        def swig_ptr_from_LongTensor(x):
            return faiss.cast_integer_to_long_ptr(
                x.storage().data_ptr() + x.storage_offset() * 8)

        N = src.size(0)
        # FIXME: not convert to cpu?
        index.add(np.ascontiguousarray(dst.cpu().numpy()))
        D = torch.empty(N, 1, device=src.device, dtype=torch.float32)
        I = torch.empty(N, 1, device=src.device, dtype=torch.float32)
        xptr = swig_ptr_from_FloatTensor(src)
        Dptr = swig_ptr_from_FloatTensor(D)
        Iptr = swig_ptr_from_LongTensor(I)
        index.search_c(N, xptr, 1, Dptr, Iptr)
        corres = torch.cat(
            (torch.arange(I.size(0)).unsqueeze(1), I.cpu().long()), 1)
        topk = D.argsort(dim=0)[:int(N * ratio)].squeeze()
        return corres[topk]
