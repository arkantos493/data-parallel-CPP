// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  constexpr std::size_t N = 42;

  queue Q{};

  int* data1 = malloc_shared<int>(N, Q);
  int* data2 = malloc_shared<int>(N, Q);

  auto e1 = Q.parallel_for(N, [=](id<1> i) { data1[i] = 1; });

  auto e2 = Q.parallel_for(N, [=](id<1> i) { data2[i] = 2; });

  auto e3 = Q.parallel_for(range{N}, {e1, e2},
                           [=](id<1> i) { data1[i] += data2[i]; });

  Q.single_task(e3, [=]() {
    for (std::size_t i = 1; i < N; ++i) {
      data1[0] += data1[i];
    }

    data1[0] /= 3;
  });
  Q.wait();

  // Check that all outputs match serial execution.
  const bool passed = data1[0] == static_cast<int>(N);
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
