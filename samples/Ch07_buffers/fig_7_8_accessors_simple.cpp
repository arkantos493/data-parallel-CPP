// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  constexpr std::size_t N = 42;

  queue Q{};

  // create 3 buffers of 42 ints
  buffer<int> A{range{N}};
  buffer<int> B{range{N}};
  buffer<int> C{range{N}};
  accessor pC{C};

  Q.submit([&](handler& h) {
    accessor aA{A, h};
    accessor aB{B, h};
    accessor aC{C, h};
    h.parallel_for(N, [=](id<1> i) {
      aA[i] = 1;
      aB[i] = 40;
      aC[i] = 0;
    });
  });
  Q.submit([&](handler& h) {
    accessor aA{A, h};
    accessor aB{B, h};
    accessor aC{C, h};
    h.parallel_for(N, [=](id<1> i) { aC[i] += aA[i] + aB[i]; });
  });
  Q.submit([&](handler& h) {
    h.require(pC);
    h.parallel_for(N, [=](id<1> i) { pC[i]++; });
  });

  // Check that all outputs match serial execution.
  host_accessor result{C};
  bool passed = true;
  for (std::size_t i = 0; i < N; i++) {
    if (result[i] != static_cast<int>(N)) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
