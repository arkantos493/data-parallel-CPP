// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  constexpr std::size_t N = 42;

  queue Q{};

  int* data = malloc_shared<int>(N, Q);

  auto e = Q.parallel_for(N, [=](id<1> i) { data[i] = 1; });

  Q.submit([&](handler& h) {
    h.depends_on(e);
    h.single_task([=]() {
      for (std::size_t i = 1; i < N; ++i) {
        data[0] += data[i];
      }
    });
  });
  Q.wait();

  // Check that all outputs match serial execution.
  const bool passed = data[0] == static_cast<int>(N);
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
