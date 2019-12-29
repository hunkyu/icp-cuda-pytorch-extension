import os
import time
import torch
import argparse
import numpy as np
from icp import icp, icp_faiss


def icp_cuda_test(xyz, new_xyz, xyz_labels, new_xyz_labels):
    print("\n=> Running icp cuda test")

    xyz = torch.cuda.FloatTensor(xyz)
    new_xyz = torch.cuda.FloatTensor(new_xyz)

    torch.cuda.synchronize()
    start_time = time.time()
    corres = icp(xyz, new_xyz, 100, threshold=1.0,
                 ratio=0.5).numpy().astype(int)
    torch.cuda.synchronize()
    end_time = time.time()
    print("    * ICP CUDA computation time: {}s".format(end_time - start_time))
    correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
    acc = correct / (xyz.size(0) * 0.5)
    print("    * Matching acc: {}".format(acc))


def icp_faiss_test(xyz, new_xyz, xyz_labels, new_xyz_labels):
    print("\n=> Running faiss test")

    torch.cuda.synchronize()
    start_time = time.time()
    corres = icp_faiss(xyz, new_xyz)
    torch.cuda.synchronize()
    end_time = time.time()
    print("    * ICP faiss computation time: {}s".format(end_time - start_time))
    correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
    acc = correct / (xyz.shape[0] * 0.5)
    print("    * Matching acc: {}".format(acc))


def load_files(seq, id):
    file_path = "data/semanticKITTI/sequences/%s/velodyne/%s.bin"
    file_path = file_path % (seq, id)
    with open(file_path, 'rb') as b:
        block = np.fromfile(b, dtype=np.float32).reshape(-1, 4)

    label_path = file_path.replace(
        '.bin', '.label').replace('velodyne', 'labels')
    with open(label_path, 'rb') as a:
        labels = np.fromfile(a, dtype=np.int32).reshape(-1)

    return block, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--num_points", type=int, default=120000)
    parser.add_argument("--gpu", type=str, default='3')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print("=> Load point clouds")
    xyz, xyz_labels = load_files('00', '000000')
    new_xyz, new_xyz_labels = load_files('00', '000001')
    if xyz.shape[0] > args.num_points:
        xyz = xyz[:args.num_points, :3]
        new_xyz = new_xyz[:args.num_points, :3]
    print("    * Point cloud 1's shape {}".format(xyz.shape))
    print("    * Point cloud 2's shape {}".format(new_xyz.shape))

    icp_faiss_test(xyz, new_xyz, xyz_labels, new_xyz_labels)
    # icp_cuda_test(xyz, new_xyz, xyz_labels, new_xyz_labels)
