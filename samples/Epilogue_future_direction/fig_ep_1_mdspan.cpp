// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <experimental/mdspan>

using namespace sycl;
namespace stdex = std::experimental;

int main() {
  constexpr int N = 4;
  constexpr int M = 2;

  queue Q{};

  int* data = malloc_shared<int>(N * M, Q);

  stdex::mdspan<int, N, M> view{data};
  Q.parallel_for(range<2>{N, M}, [=](id<2> idx) {
     const std::size_t i = idx[0];
     const std::size_t j = idx[1];
     view(i, j) = static_cast<int>(i * M + j);
   }).wait();

  bool passed = true;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < M; ++j) {
      if (data[i * M + j] != static_cast<int>(i * M + j)) {
        passed = false;
      }
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';

  free(data, Q);
  return passed ? 0 : 1;
}
