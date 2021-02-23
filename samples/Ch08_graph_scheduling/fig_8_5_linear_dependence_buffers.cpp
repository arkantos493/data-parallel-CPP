// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  constexpr std::size_t N = 42;

  queue Q{};

  buffer<int> data{range{N}};

  Q.submit([&](handler& h) {
    accessor a{data, h};
    h.parallel_for(N, [=](id<1> i) { a[i] = 1; });
  });

  Q.submit([&](handler& h) {
    accessor a{data, h};
    h.single_task([=]() {
      for (std::size_t i = 1; i < N; ++i) {
        a[0] += a[i];
      }
    });
  });

  host_accessor h_a{data};
  // Check that all outputs match serial execution.
  const bool passed = h_a[0] == static_cast<int>(N);
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
