#include <ATen/TensorUtils.h>
#include <omp.h>
#include <stdio.h>
#include "nnSearch.h"

template <typename T>
void nnSearch(const T *query, const T *ref, const int N, const int M, T *dist) {
#pragma omp parallel
  {
    for (int i = omp_get_thread_num(); i < N; i += omp_get_num_threads()) {
      float minDist = INFINITY;
      int minIdx = -1;

      float queryX = query[i * 3 + 0];
      float queryY = query[i * 3 + 1];
      float queryZ = query[i * 3 + 2];

      float refX, refY, refZ, tempDist;

      for (int j = 0; j < M; j++) {
        refX = ref[j * 3 + 0];
        refY = ref[j * 3 + 1];
        refZ = ref[j * 3 + 2];

        tempDist = (queryX - refX) * (queryX - refX) +
                   (queryY - refY) * (queryY - refY) +
                   (queryZ - refZ) * (queryZ - refZ);
        if (tempDist < minDist) {
          minDist = tempDist;
          minIdx = j;
        }
      }  // forj

      dist[i * 2] = minIdx;
      dist[i * 2 + 1] = minDist;
    }  // fori
  }
}

at::Tensor nnSearch_cpu(const at::Tensor &query, const at::Tensor &ref) {
  AT_ASSERTM(query.device().is_cpu(), "query point cloud must be a CPU tensor");
  AT_ASSERTM(ref.device().is_cpu(), "ref point cloud must be a CPU tensor");
  at::TensorArg query_t{query, "query", 1}, ref_t{ref, "ref", 2};

  at::CheckedFrom c = "nnSearch_cpu";  // function name for check
  at::checkAllSameType(c, {query_t, ref_t});

  auto N = query.size(0);
  auto M = ref.size(0);

  auto dist = at::empty({N, 2}, query.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "nnSearch", [&] {
    nnSearch<scalar_t>(query.contiguous().data_ptr<scalar_t>(),
                       ref.contiguous().data_ptr<scalar_t>(), N, M,
                       dist.contiguous().data_ptr<scalar_t>());
  });

  return dist;
}
