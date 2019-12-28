import os
import time
import torch
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from icp import icp


def icp_test(xyz, new_xyz, radius):
    print("\n=> Running icp test")

    xyz = torch.cuda.FloatTensor(xyz)
    new_xyz = torch.cuda.FloatTensor(new_xyz)

    torch.cuda.synchronize()
    start_time = time.time()
    icp(xyz, new_xyz, radius, 200)
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
    parser.add_argument("--num_points", type=int, default=60000)
    parser.add_argument("--gpu", type=str, default='3')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print("=> Load point clouds")
    xyz = load_files('00', '000000')[:, :3]
    new_xyz = load_files('00', '000001')[:, :3]
    if xyz.shape[0] > args.num_points:
        xyz = xyz[:args.num_points, :]
        new_xyz = new_xyz[:args.num_points, :]
    print("    * Point cloud 1's shape {}".format(xyz.shape))
    print("    * Point cloud 2's shape {}".format(new_xyz.shape))

    icp_test(xyz, new_xyz, args.radius)
