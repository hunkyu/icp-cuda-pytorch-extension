#pragma once
#include <torch/types.h>

/**
 * TODO: batch
 * C interface for point cloud nearest neighbor search
 * @param query query point cloud with size Nx3
 * @param ref reference point cloud with size Nx3
 * @return tensor with size Nx2, where the first column is index, distance and
 * the second column is distance
 */
at::Tensor nnSearch(const at::Tensor& src, const at::Tensor& dst);

// TODO: KDTree version
at::Tensor nnSearchKdtree(const at::Tensor& src, const at::Tensor& dst);

// Python interface
inline at::Tensor nnSearch(const at::Tensor& src, const at::Tensor& dst) {
  if (src.type().is_cuda()) {
#ifdef WITH_CUDA
    return nnSearch(src, dst);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not implmented");
}

inline at::Tensor nnSearchKdtree(const at::Tensor& src, const at::Tensor& dst) {
  if (src.type().is_cuda()) {
#ifdef WITH_CUDA
    return nnSearchKdtree(src, dst);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not implmented");
}