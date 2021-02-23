// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
using namespace sycl;

int main() {
  constexpr int N = 42;
  std::array<int, N> a{};
  std::array<int, N> b{};
  std::array<int, N> c{};

  queue Q{};

  // We will learn how to simplify this example later
  buffer A{a};
  buffer B{b};
  buffer C{c};

  Q.submit([&](handler& h) {
    accessor accA{A, h, read_only};
    accessor accB{B, h, write_only};
    h.parallel_for(  // computeB
        N, [=](id<1> i) { accB[i] = accA[i] + 1; });
  });

  Q.submit([&](handler& h) {
    accessor accA{A, h, read_only};
    h.parallel_for(  // readA
        N, [=](id<1> i) {
          // Useful only as an example
          [[maybe_unused]] int data = accA[i];
        });
  });

  Q.submit([&](handler& h) {
    // RAW of buffer B
    accessor accB{B, h, read_only};
    accessor accC{C, h, write_only};
    h.parallel_for(  // computeC
        N, [=](id<1> i) { accC[i] = accB[i] + 2; });
  });

  // read C on host
  host_accessor host_accC{C, read_only};
  for (std::size_t i = 0; i < N; ++i) {
    std::cout << host_accC[i] << ' ';
  }
  std::cout << '\n';

  // test results
  for (std::size_t i = 0; i < N; ++i) {
    if (host_accC[i] != 3) {
      std::cout << "Wrong results\n";
      return 1;
    }
  }
  std::cout << "Correct results\n";
  return 0;
}
