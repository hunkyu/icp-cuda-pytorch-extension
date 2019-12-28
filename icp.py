import torch
import custom_ext as _C


def icp(src, dst, radius, maxIter, threshold=0.005):
    prev_err = 0

    for i in range(maxIter):
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
        err = torch.sqrt(torch.norm(src - dst_temp, dim=1)).mean()
        if torch.abs(err - prev_err) < threshold:
            prev_err = err
            break
        else:
            prev_err = err

    return idx
