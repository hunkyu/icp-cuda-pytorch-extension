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
                         const int M, int *idx) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    float minDist = INFINITY;
    int minIdx = -1;

    float srcX = src[index * 3 + 0];
    float srcY = src[index * 3 + 1];
    float srcZ = src[index * 3 + 2];

    float dstX, dstY, dstZ, tempDist;

    for (int j = 0; j < M; j++) {
      dstX = dst[j * 3 + 0];
      dstY = dst[j * 3 + 1];
      dstZ = dst[j * 3 + 2];

      tempDist = (srcX - dstX) * (srcX - dstX) + (srcY - dstY) * (srcY - dstY) +
                 (srcZ - dstZ) * (srcZ - dstZ);
      if (tempDist < minDist) {
        minDist = tempDist;
        minIdx = j;
      }
    } // forj

    idx[index] = minIdx;
  }
}

at::Tensor nnSearch(const at::Tensor &src, const at::Tensor &dst) {
  AT_ASSERTM(src.device().is_cuda(), "src point cloud must be a CUDA tensor");
  AT_ASSERTM(dst.device().is_cuda(), "dst point cloud must be a CUDA tensor");
  at::TensorArg src_t{src, "src", 1}, dst_t{dst, "dst", 2};

  at::CheckedFrom c = "ICP_cuda"; // function name for check
  at::checkAllSameGPU(c, {src_t, dst_t});
  at::checkAllSameType(c, {src_t, dst_t});
  at::cuda::CUDAGuard device_guard(src.device());

  auto N = src.size(0);
  auto M = dst.size(0);

  auto idx = at::empty({N}, src.options()).to(at::kInt);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(at::cuda::ATenCeilDiv(N, 512L), 4096L));
  dim3 block(512);

  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "nnSearch", [&] {
    nnSearch<scalar_t><<<grid, block, 0, stream>>>(
        N, src.contiguous().data_ptr<scalar_t>(),
        dst.contiguous().data_ptr<scalar_t>(), M,
        idx.contiguous().data_ptr<int>());
  });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  return idx;
}
