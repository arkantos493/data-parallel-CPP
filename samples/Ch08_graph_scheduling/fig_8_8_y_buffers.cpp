// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  constexpr std::size_t N = 42;

  queue Q{};

  buffer<int> data1{range{N}};
  buffer<int> data2{range{N}};

  Q.submit([&](handler& h) {
    accessor a{data1, h};
    h.parallel_for(N, [=](id<1> i) { a[i] = 1; });
  });

  Q.submit([&](handler& h) {
    accessor b{data2, h};
    h.parallel_for(N, [=](id<1> i) { b[i] = 2; });
  });

  Q.submit([&](handler& h) {
    accessor a{data1, h};
    accessor b{data2, h, read_only};
    h.parallel_for(N, [=](id<1> i) { a[i] += b[i]; });
  });

  Q.submit([&](handler& h) {
    accessor a{data1, h};
    h.single_task([=]() {
      for (std::size_t i = 1; i < N; ++i) a[0] += a[i];

      a[0] /= 3;
    });
  });

  host_accessor h_a{data1};
  // Check that all outputs match serial execution.
  const bool passed = h_a[0] == static_cast<int>(N);
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
