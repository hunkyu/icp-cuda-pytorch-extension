#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                   \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void nnSearch(const int nthreads, const T *src, const T *dst,
                         const int N, const int M, int *idx) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {} // CUDA_1D_KERNEL_LOOP
} // nnSearch

at::Tensor ICP_cuda(const at::Tensor &src, const at::Tensor &dst,
                    const float radius, const int max_iter) {
  AT_ASSERTM(src.device().is_cuda(), "src point cloud must be a CUDA tensor");
  AT_ASSERTM(dst.device().is_cuda(), "dst point cloud must be a CUDA tensor");
  at::TensorArg src_t{src, "src", 1}, dst_t{dst, "dst", 2};

  // A string describing which function did checks on its input arguments.
  at::CheckedFrom c = "ICP_cuda";
  at::checkAllSameGPU(c, {src_t, dst_t});
  at::checkAllSameType(c, {src_t, dst_t});
  at::cuda::CUDAGuard device_guard(src.device());

  auto N = src.size(0);
  auto M = dst.size(0);

  auto idx = at::empty({N, 2}, src.options()).to(at::kInt);
  auto nThreads = N * 2;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(at::cuda::ATenCeilDiv(nThreads, 512L), 4096L));
  dim3 block(512);

  nnSearch<<<grid, block, 0, stream>>>(nThreads, src.contiguous().data<float>(),
                                       dst.contiguous().data<float>(), N, M,
                                       idx.contiguous().data<int>());

  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "nnSearch", [&] {
    nnSearch<scalar_t><<<grid, block, 0, stream>>>(
        nThreads, src.contiguous().data<scalar_t>(),
        dst.contiguous().data<scalar_t>(), N, M, idx.contiguous().data<int>());
  });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return idx;
}
