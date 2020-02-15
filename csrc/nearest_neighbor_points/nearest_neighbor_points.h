// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#pragma once
#include <torch/extension.h>
#include "pytorch3d_cutils.h"

#ifdef WITH_CUDA
/**
 * Compute indices of nearest neighbors in pointcloud p2
 * to points in pointcloud p1.
 * @param p1 FloatTensor of shape (N, P1, D) giving a batch of pointclouds
 *           each containing P1 points of dimension D.
 * @param p2 FloatTensor of shape (N, P2, D) giving a batch of pointclouds
 *           each containing P2 points of dimension D.
 * @return LongTensor of shape (N, P1), where p1_neighbor_idx[n, i] = j
 *         means that the nearest neighbor to p1[n, i] in the cloud
 *         p2[n] is p2[n, j].
 */
std::pair<at::Tensor, at::Tensor> NearestNeighborIdxCuda(at::Tensor p1, at::Tensor p2);
#endif

/**
 * Compute indices of nearest neighbors in pointcloud p2
 * to points in pointcloud p1.
 * @param p1 FloatTensor of shape (N, P1, D) giving a batch of pointclouds
 *           each containing P1 points of dimension D.
 * @param p2 FloatTensor of shape (N, P2, D) giving a batch of pointclouds
 *           each containing P2 points of dimension D.
 * @return p1_neighbor_idx, LongTensor of shape (N, P1, 2),
 *         where p1_neighbor_idx[n, i, 0] = j means that the
 *         nearest neighbor to p1[n, i] in the cloud p2[n] is p2[n, j],
 *         and p1_neighbor_idx[n, i, 1] means the distance
 */
std::pair<at::Tensor, at::Tensor> NearestNeighborIdxCpu(at::Tensor p1,
                                                        at::Tensor p2);

// Python interface
std::pair<at::Tensor, at::Tensor> NearestNeighborIdx(at::Tensor p1,
                                                     at::Tensor p2) {
  if (p1.type().is_cuda() && p2.type().is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(p1);
    CHECK_CONTIGUOUS_CUDA(p2);
    return NearestNeighborIdxCuda(p1, p2);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return NearestNeighborIdxCpu(p1, p2);
};