#pragma once
#include <torch/types.h>

at::Tensor ICPCuda(const at::Tensor& src, const at::Tensor& dst,
                   const float radius, const int maxIter);

// Python interface
inline at::Tensor ICP(const at::Tensor& src, const at::Tensor& dst,
                      const float radius, const int maxIter) {
  if (src.type().is_cuda()) {
#ifdef WITH_CUDA
    return ICPCuda(src, dst, radius, maxIter);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not implmented");
}
