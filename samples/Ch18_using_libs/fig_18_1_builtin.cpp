// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
using namespace sycl;

int main() {
  constexpr std::size_t size = 9;
  std::array<double, size> A{};
  std::iota(A.begin(), A.end(), 0);
  std::array<double, size> B{};
  std::iota(B.begin(), B.end(), 0);

  bool pass = true;

  queue Q{};

  range sz{size};

  buffer bufA{A};
  buffer bufB{B};
  buffer<bool> bufP{&pass, 1};

  Q.submit([&](handler& h) {
    accessor accA{bufA, h};
    accessor accB{bufB, h};
    accessor accP{bufP, h};

    h.parallel_for(size, [=](id<1> idx) {
      accA[idx] = sycl::log(accA[idx]);
      accB[idx] = sycl::log(accB[idx]);
      if (!sycl::isequal(accA[idx], accB[idx])) {
        accP[0] = false;
      }
    });
  });

  host_accessor host_A{bufA};
  host_accessor host_P{bufP};
  const bool passed = host_P[0] && host_A[4] == std::log(4.0);
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
