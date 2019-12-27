#pragma once
#include <torch/types.h>

at::Tensor ICP_cuda(const at::Tensor& src, const at::Tensor& dst,
                    const float radius, const int max_iter);

// Python interface
inline at::Tensor ICP(const at::Tensor& src, const at::Tensor& dst,
                      const float radius, const int max_iter) {
  if (src.type().is_cuda()) {
#ifdef WITH_CUDA
    return ICP_cuda(src, dst, radius, max_iter);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not implmented");
}
