// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
using namespace sycl;

using matrix_type = float;
extern const std::size_t matrixSize;

// This function must be implemented for each sample:
template <typename T>
double run_sycl(const std::vector<T>& vecA, const std::vector<T>& vecB,
                std::vector<T>& vecC);

template <typename T>
static std::vector<T> make_random_square_matrix() {
  std::default_random_engine gen(42);
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  auto rng = [&]() { return dist(gen); };

  std::vector<T> matrix(matrixSize * matrixSize);
  std::generate(matrix.begin(), matrix.end(), rng);
  return matrix;
}

template <typename T>
static void compute_reference(const std::vector<T>& matrixA,
                              const std::vector<T>& matrixB,
                              std::vector<T>& matrixC) {
  const std::size_t M = matrixSize;
  const std::size_t N = matrixSize;
  const std::size_t K = matrixSize;

  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t n = 0; n < N; ++n) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += matrixA[m * K + k] * matrixB[k * N + n];
      }
      matrixC[m * N + n] = sum;
    }
  }
}

template <typename T>
int check_results(const std::vector<T>& matrixC,
                  const std::vector<T>& referenceC) {
  const std::size_t M = matrixSize;
  const std::size_t N = matrixSize;

  float err = 0.0f;
  for (std::size_t i = 0; i < M * N; ++i) {
    float localErr = std::fabs(matrixC[i] - referenceC[i]) /
                     std::max(std::fabs(matrixC[i]), std::fabs(referenceC[i]));
    err = std::max(localErr, err);
    if (localErr >= 0.001f) {
      std::cerr << "Error at index " << i << ": Wanted " << referenceC[i]
                << ", got " << matrixC[i] << std::endl;
      break;
    }
  }

  return err < 0.001f;
}

int main() {
  std::vector<matrix_type> matrixA = make_random_square_matrix<matrix_type>();
  std::vector<matrix_type> matrixB = make_random_square_matrix<matrix_type>();
  std::vector<matrix_type> referenceC(matrixSize * matrixSize, 0);
  compute_reference(matrixA, matrixB, referenceC);

  std::vector<matrix_type> matrixC(matrixSize * matrixSize, 0);
  auto seconds = run_sycl(matrixA, matrixB, matrixC);

  if (!check_results(matrixC, referenceC)) {
    std::cout << "Wrong results\n";
    return 1;
  }

  auto gflops = static_cast<double>(matrixSize) *
                static_cast<double>(matrixSize) *
                (static_cast<double>(matrixSize) +   // multiplications
                 static_cast<double>(matrixSize)) /  // additions
                seconds /
                1e9;
  std::cout << "Correct results!\n";
  std::cout << "GFlops: " << gflops << std::endl;

  return 0;
}
