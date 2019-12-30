import os
import time
import torch
import argparse
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


def icp_cpu_test(xyz, new_xyz, xyz_labels, new_xyz_labels):
    torch.cuda.empty_cache()
    print("\n=> Running ICP CPU test")

    xyz = torch.FloatTensor(xyz)
    new_xyz = torch.FloatTensor(new_xyz)

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        corres = icp(xyz, new_xyz, 15, ratio=0.5).numpy().astype(int)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    torch.cuda.synchronize()
    end_time = time.time()
    print("    * ICP CUDA computation time: {}s".format(end_time - start_time))
    correct = (xyz_labels[corres[:, 0]] == new_xyz_labels[corres[:, 1]]).sum()
    acc = correct / corres.shape[0]
    print("    * Matching acc: {}".format(acc))


def icp_cuda_test(xyz, new_xyz, xyz_labels, new_xyz_labels):
    torch.cuda.empty_cache()
    print("\n=> Running ICP CUDA test")

    xyz = torch.cuda.FloatTensor(xyz)
    new_xyz = torch.cuda.FloatTensor(new_xyz)

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        corres = icp(xyz, new_xyz, 15, ratio=0.5).numpy().astype(int)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
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


def nn_search_faiss_test(xyz, new_xyz, xyz_labels, new_xyz_labels):
    print("\n=> Running Nearest Neighbor search faiss test")

    torch.cuda.synchronize()
    start_time = time.time()
    corres = nn_search_faiss(xyz, new_xyz)
    torch.cuda.synchronize()
    end_time = time.time()
    print("    * NN search faiss computation time: {}s".format(end_time - start_time))
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
    print("    * Point cloud 1's shape {}".format(xyz.shape))
    print("    * Point cloud 2's shape {}".format(new_xyz.shape))

    nn_search_cpu_test(xyz[:, :3], new_xyz[:, :3], xyz_labels, new_xyz_labels)
    # icp_cpu_test(xyz[:, :3], new_xyz[:, :3], xyz_labels, new_xyz_labels)
    nn_search_cuda_test(xyz[:, :3], new_xyz[:, :3], xyz_labels, new_xyz_labels)
    icp_cuda_test(xyz[:, :3], new_xyz[:, :3], xyz_labels, new_xyz_labels)
