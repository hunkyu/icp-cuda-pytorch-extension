# ICP CUDA PyTorch Extension

CUDA Iterative Closest Point algorithm for PyTorch (>=1.3).

## Install

```
$ python setup.py install
```

## Benchmark

For ICP, the most time consuming part is NN search. And the CUDA version of SVD and matrix multiplication is not faster than the CPU version for small matrix. Thus I only implemented CUDA version's NN search, and leave the other parts of ICP CPU.

Because Open3D uses KDTree to accelerate NN search, the CUDA accleration is not very significant (only 2x faster).

Benchmark for 80000 points matching with single NVIDIA GTX 2080Ti and Xeon(R) CPU E5-2678 v3 @ 2.50GHz is given below. You can run the benchmark by

```
$ python ctest.py --num_points=80000
```

|      | Nearest Neighbor Search      | Iterative Closest Point      |
| ---- | ---------------------------- | ---------------------------- |
| CPU  | acc = 93.97%, time = 1.06s   | acc = 97.65%, time = 27.96ms |
| CUDA | acc = 93.97%, time = 29.1ms  | acc = 97.92%, time = 14.34ms |

## Todo

- [x] Achieve the same matching accuracy as Open3D CPU version
- [x] Accelerate CUDA ICP
- [ ] Use KDTree for NN search