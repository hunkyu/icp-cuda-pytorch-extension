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
                         const int M, T *dstTemp) {
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

    dstTemp[index] = dst[minIdx * 3 + 0];
    dstTemp[index] = dst[minIdx * 3 + 1];
    dstTemp[index] = dst[minIdx * 3 + 2];
  }
}

at::Tensor ICPCuda(const at::Tensor &src, const at::Tensor &dst,
                   const float radius, const int maxIter) {
  AT_ASSERTM(src.device().is_cuda(), "src point cloud must be a CUDA tensor");
  AT_ASSERTM(dst.device().is_cuda(), "dst point cloud must be a CUDA tensor");
  at::TensorArg src_t{src, "src", 1}, dst_t{dst, "dst", 2};

  at::CheckedFrom c = "ICP_cuda"; // function name for check
  at::checkAllSameGPU(c, {src_t, dst_t});
  at::checkAllSameType(c, {src_t, dst_t});
  at::cuda::CUDAGuard device_guard(src.device());

  auto N = src.size(0);
  auto M = dst.size(0);

  auto idx = at::empty({N, 2}, src.options()).to(at::kInt);
  auto dstTemp = at::zeros_like(src);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(at::cuda::ATenCeilDiv(N, 512L), 4096L));
  dim3 block(512);

  for (int i = 0; i < maxIter; i++) {
    // 1. Find Nearest Neighbor
    AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "nnSearch", [&] {
      nnSearch<scalar_t><<<grid, block, 0, stream>>>(
          N, src.contiguous().data_ptr<scalar_t>(),
          dst.contiguous().data_ptr<scalar_t>(), M,
          dstTemp.contiguous().data_ptr<scalar_t>());
    });
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());

    // 2. Compute H
    auto srcCenter = at::mean(src, 0);
    auto dstTempCenter = at::mean(dstTemp, 0);
    auto srcNorm = at::sub(src, srcCenter);      // Nx3
    auto dstNorm = at::sub(dst, dstTempCenter);  // Nx3
    auto hMatrix = at::mm(srcNorm.t(), dstNorm); // 3x3

    // 3. SVD
    auto out = at::svd(hMatrix);
    auto U = std::get<0>(out);
    auto S = std::get<1>(out);
    auto V = std::get<2>(out);

    // 4. Rotation Matrix and Translation Vector
    auto R = at::mul(U, V);
    auto t = dstCenter - at::mul(R, srcCenter); // TODO: check size ?
  }
  return idx;
}
