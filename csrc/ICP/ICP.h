#pragma once
#include <torch/types.h>

/**
 * TODO: batch
 * CUDA interface for point cloud nearest neighbor search
 * @param query query point cloud with size Nx3
 * @param ref reference point cloud with size Nx3
 * @return tensor with size Nx2, where the first column is index, distance and
 * the second column is distance
 */
at::Tensor ICP(const at::Tensor& query, const at::Tensor& ref,
               const int maxIter);

// Python interface
inline at::Tensor ICP(const at::Tensor& query, const at::Tensor& ref,
                      const int maxIter) {
  if (src.type().is_cuda()) {
#ifdef WITH_CUDA
    return ICP(src, dst, maxIter);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not implmented");
}
