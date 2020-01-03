#include "nnSearch/nnSearch.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>

at::Tensor ICP_cuda(const at::Tensor &query, const at::Tensor &ref,
                    const int maxIter) {
  AT_ASSERTM(query.device().is_cuda(),
             "query point cloud must be a CUDA tensor");
  AT_ASSERTM(ref.device().is_cuda(), "ref point cloud must be a CUDA tensor");
  at::TensorArg query_t{query, "query", 1}, ref_t{ref, "ref", 2};

  at::CheckedFrom c = "ICP_cuda"; // function name for check
  at::checkAllSameGPU(c, {query_t, ref_t});
  at::checkAllSameType(c, {query_t, ref_t});
  at::cuda::CUDAGuard device_guard(query.device());

  auto N = query.size(0);
  auto M = ref.size(0);

  auto dist = at::empty({N, 2}, query.options());
  auto refTemp = at::empty({N, 3}, query.options());
  auto refIdxs = at::empty({N}, at::kLong);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  for (int i = 0; i < maxIter; i++) {
    refIdxs = std::get<0>(nnSearch_cuda(query, ref));
    std::cout << typeid(refIdxs).name() << std::endl;
    refTemp = at::gather(ref, 0, refIdxs);
  }

  return dist;
}