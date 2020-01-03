#pragma once
#include <torch/types.h>

#ifdef WITH_CUDA
/**
 * TODO: batch
 * CUDA interface for point cloud nearest neighbor search
 * @param query query point cloud with size Nx3
 * @param ref reference point cloud with size Nx3
 * @return tensor with size Nx2, where the first column is index, distance and
 * the second column is distance
 */
at::Tensor ICP_cuda(const at::Tensor& query, const at::Tensor& ref,
                    const int maxIter);
#endif

// Python interface
inline at::Tensor ICP(const at::Tensor& query, const at::Tensor& ref,
                      const int maxIter) {
  if (query.type().is_cuda()) {
#ifdef WITH_CUDA
    return ICP_cuda(query, ref, maxIter);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not implmented");
}
