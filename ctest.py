import os
import time
import torch
import argparse
import unittest
import numpy as np
import open3d as o3d

from layers import *

# TODO: abstract unit-test


def icp_open3d_cpu_test(xyz, new_xyz, xyz_labels, new_xyz_labels):
    print("\n=> Running ICP Open3D CPU test")

    cur_3d = o3d.geometry.PointCloud()
    prev_3d = o3d.geometry.PointCloud()
    cur_3d.points = o3d.Vector3dVector(xyz.astype(np.float32))
    prev_3d.points = o3d.Vector3dVector(new_xyz.astype(np.float32))

    start_time = time.time()
    registration = o3d.registration_icp(
        cur_3d, prev_3d, 1.0,
        criteria=o3d.registration.ICPConvergenceCriteria(
            max_iteration=20))
    end_time = time.time()
    corres = np.asarray(registration.correspondence_set)
    print("    * ICP Open3D CPU computation time: {}s".format(end_time - start_time))
    correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
    ratio = corres.shape[0] / xyz.shape[0]
    acc = correct / corres.shape[0]
    print("    * Matching ratio: {}".format(ratio))
    print("    * Matching acc: {}".format(acc))


def icp_pytorch_cuda_test(xyz, new_xyz, xyz_labels, new_xyz_labels):
    torch.cuda.empty_cache()
    print("\n=> Running ICP (PyTorch) all CUDA test")

    start_time = time.time()
    xyz = torch.cuda.FloatTensor(xyz)
    new_xyz = torch.cuda.FloatTensor(new_xyz)
    end_time = time.time()
    torch.cuda.synchronize()
    print("    * Load data time: {}s".format(end_time - start_time))

    torch.cuda.synchronize()
    start_time = time.time()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    corres = icp_pytorch(xyz, new_xyz, 15, ratio=0.5).numpy().astype(int)
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    torch.cuda.synchronize()
    end_time = time.time()
    print("    * ICP CUDA computation time: {}s".format(end_time - start_time))
    correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
    acc = correct / corres.shape[0]
    print("    * Matching acc: {}".format(acc))


def icp_pytorch_test(xyz, new_xyz, xyz_labels, new_xyz_labels):
    torch.cuda.empty_cache()
    print("\n=> Running ICP (PyTorch) test")

    xyz = torch.FloatTensor(xyz)
    new_xyz = torch.FloatTensor(new_xyz)

    torch.cuda.synchronize()
    start_time = time.time()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    corres = icp_pytorch(xyz, new_xyz, 15, ratio=0.5).numpy().astype(int)
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    torch.cuda.synchronize()
    end_time = time.time()
    print("    * ICP CUDA computation time: {}s".format(end_time - start_time))
    correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
    acc = correct / corres.shape[0]
    print("    * Matching acc: {}".format(acc))


def nn_search_cpu_test(xyz, new_xyz, xyz_labels, new_xyz_labels):
    torch.cuda.empty_cache()
    print("\n=> Running Nearest Neighbor search CPU test")

    xyz = torch.FloatTensor(xyz)
    new_xyz = torch.FloatTensor(new_xyz)

    torch.cuda.synchronize()
    start_time = time.time()
    corres = nn_search(xyz, new_xyz, ratio=0.5).numpy().astype(int)
    torch.cuda.synchronize()
    end_time = time.time()
    print("    * NN search computation time: {}s".format(end_time - start_time))
    correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
    acc = correct / corres.shape[0]
    print("    * Matching acc: {}".format(acc))


def nn_search_cuda_test(xyz, new_xyz, xyz_labels, new_xyz_labels):
    torch.cuda.empty_cache()
    print("\n=> Running Nearest Neighbor search CUDA test")

    xyz = torch.cuda.FloatTensor(xyz)
    new_xyz = torch.cuda.FloatTensor(new_xyz)

    torch.cuda.synchronize()
    start_time = time.time()
    corres = nn_search(xyz, new_xyz, ratio=0.5).numpy().astype(int)
    torch.cuda.synchronize()
    end_time = time.time()
    print("    * NN search computation time: {}s".format(end_time - start_time))
    correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
    acc = correct / corres.shape[0]
    print("    * Matching acc: {}".format(acc))


def batch_nn_search_test(xyz, new_xyz, xyz_labels, new_xyz_labels):
    torch.cuda.empty_cache()
    print("\n=> Running Batch Nearest Neighbor search test")

    xyz = torch.cuda.FloatTensor(xyz)
    new_xyz = torch.cuda.FloatTensor(new_xyz)

    torch.cuda.synchronize()
    start_time = time.time()
    corres = batch_nn_search(xyz, new_xyz, ratio=0.5).numpy().astype(int)
    torch.cuda.synchronize()
    end_time = time.time()
    print("    * Batch NN search computation time: {}s".format(end_time - start_time))
    for b in range(corres.shape[0]):
        correct = (xyz_labels[b, corres[b, :, 0]] ==
                   new_xyz_labels[b, corres[b, :, 1]]).sum()
        acc = correct / corres.shape[1]
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


def load_files_seq(seq, start_id, length, num_points):
    block_lists = []
    label_lists = []

    for id in range(start_id, start_id+length):
        file_path = "data/semanticKITTI/sequences/%s/velodyne/%06d.bin"
        file_path = file_path % (seq, id)
        with open(file_path, 'rb') as b:
            block = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        block_lists.append(block[:num_points, :3])

        label_path = file_path.replace(
            '.bin', '.label').replace('velodyne', 'labels')
        with open(label_path, 'rb') as a:
            labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        label_lists.append(labels[:num_points])

    return np.array(block_lists), np.array(label_lists)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--num_points", type=int, default=50000)
    parser.add_argument("--gpu", type=str, default='3')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print("=> Load point clouds")
    xyz, xyz_labels = load_files('00', '000000')
    new_xyz, new_xyz_labels = load_files('00', '000001')
    if xyz.shape[0] > args.num_points:
        xyz = xyz[:args.num_points, :3]
        new_xyz = new_xyz[:args.num_points, :3]
        xyz_labels = xyz_labels[:args.num_points]
        new_xyz_labels = new_xyz_labels[:args.num_points]
    print("    * Point cloud 1's shape {}".format(xyz.shape))
    print("    * Point cloud 2's shape {}".format(new_xyz.shape))

    nn_search_cuda_test(xyz[:, :3], new_xyz[:, :3], xyz_labels, new_xyz_labels)
    # nn_search_cpu_test(xyz[:, :3], new_xyz[:, :3], xyz_labels, new_xyz_labels)
    # icp_pytorch_cuda_test(xyz[:, :3], new_xyz[:, :3],
    #                       xyz_labels, new_xyz_labels)
    # icp_pytorch_test(xyz[:, :3], new_xyz[:, :3], xyz_labels, new_xyz_labels)
    # icp_open3d_cpu_test(xyz[:, :3], new_xyz[:, :3], xyz_labels, new_xyz_labels)

    print("=> Load point cloud sequences")
    xyzs, xyzs_labels = load_files_seq('00', 0, 5, args.num_points)
    new_xyzs, new_xyzs_labels = load_files_seq('00', 1, 5, args.num_points)
    print("    * Point cloud sequence 1's shape {}".format(xyzs.shape))
    print("    * Point cloud sequence 2's shape {}".format(new_xyzs.shape))
    batch_nn_search_test(xyzs, new_xyzs, xyzs_labels, new_xyzs_labels)
