#pragma once
#include <stdexcept>
#include <algorithm>
#include "../tensor.hpp"

inline void matmul_f32_nn(
  const float* A, const float* B, float* C,
  int M, int K, int N
) {
  if (!A || !B || !C) throw std::runtime_error("matmul_f32_nn: null pointer");
  if (M <= 0 || K <= 0 || N <= 0) throw std::runtime_error("matmul_f32_nn: non-positive dims");

  const Size MN = (Size)M * (Size)N;
  for (Size idx = 0; idx < MN; idx++) C[idx] = 0.0f;

  for (int i = 0; i < M; i++) {
    const float* Arow = A + (Size)i * (Size)K;
    float* Crow = C + (Size)i * (Size)N;

    for (int k = 0; k < K; k++) {
      const float a_ik = Arow[k];
      const float* Brow = B + (Size)k * (Size)N;

      for (int j = 0; j < N; j++) {
        Crow[j] += a_ik * Brow[j];
      }
    }
  }
}

inline std::shared_ptr<Tensor> matmul2d_cpu_kernel(
  const std::shared_ptr<Tensor>& A,
  const std::shared_ptr<Tensor>& B,
  bool blocked = false
) {
  if (!A || !B) throw std::runtime_error("matmul2d_cpu_kernel: null tensor");
  if (!A->is_contiguous() || !B->is_contiguous()) throw std::runtime_error("matmul2d_cpu_kernel: requires contiguous");
  if (A->shape.size() != 2 || B->shape.size() != 2) throw std::runtime_error("matmul2d_cpu_kernel: expects 2D tensors");

  const int M = A->shape[0];
  const int K = A->shape[1];
  const int K2 = B->shape[0];
  const int N = B->shape[1];
  if (K != K2) throw std::runtime_error("matmul2d_cpu_kernel: shape mismatch");

  vector_float out((Size)M * (Size)N, 0.0f);

  const float* Ap = A->data.data();
  const float* Bp = B->data.data();
  float* Cp = out.data();

  if (blocked) matmul_f32_nn_blocked(Ap, Bp, Cp, M, K, N);
  else         matmul_f32_nn(Ap, Bp, Cp, M, K, N);

  return std::make_shared<Tensor>(std::move(out), vector_int{M, N}, false);
}
