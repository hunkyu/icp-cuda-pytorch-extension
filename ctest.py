import os
import time
import torch
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import custom_ext as _C


# TODO: remove me
os.environ['CUDA_VISIBL_DEIVCES'] = '3'


def icp_test(xyz, new_xyz, radius):
    print("\n=> Running icp test")

    xyz = torch.tensor(xyz).cuda()
    new_xyz = torch.tensor(new_xyz).cuda()

    torch.cuda.synchronize()
    start_time = time.time()
    _C.icp(xyz, new_xyz, radius, 100)
    torch.cuda.synchronize()
    end_time = time.time()
    print("    * GPU computation time: {}s".format(end_time - start_time))


def load_files(seq, id):
    file_path = "data/semanticKITTI/sequences/%s/velodyne/%s.bin"
    file_path = file_path % (seq, id)
    with open(file_path, 'rb') as b:
        block = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
    return block


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--num_points", type=int, default=100000)
    args = parser.parse_args()

    print("=> Load point clouds")
    xyz = load_files('00', '000000')[:, :3] / args.voxel_size
    new_xyz = load_files('00', '000001')[:, :3] / args.voxel_size
    if xyz.shape[0] > args.num_points:
        xyz = xyz[:args.num_points, :]
        new_xyz = new_xyz[:args.num_points, :]
    print("    * Point cloud 1's shape {}".format(xyz.shape))
    print("    * Point cloud 2's shape {}".format(new_xyz.shape))

    icp_test(xyz, new_xyz, args.radius)
