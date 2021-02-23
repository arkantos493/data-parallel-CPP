// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
#include <cmath>
#include <iostream>
using namespace sycl;

int main() {
  constexpr std::size_t size = 8;

  std::array<float, size> fpData{};
  std::iota(fpData.begin(), fpData.end(), 0.0f);
  std::array<float4, size> fp4Data{};
  for (std::size_t i = 0; i < size; ++i) {
    const float b = 4.0f * static_cast<float>(i);
    fp4Data[i] = float4(b, b + 1, b + 2, b + 3);
  }

  buffer fpBuf{fpData};
  buffer fp4Buf{fp4Data};

  queue Q{};
  Q.submit([&](handler& h) {
    accessor a{fpBuf, h};
    accessor b{fp4Buf, h};

    // BEGIN CODE SNIP
    h.parallel_for(8, [=](id<1> i) {
      const float x = a[i];
      const float4 y4 = b[i];
      a[i] = x + sycl::length(y4);
    });
    // END CODE SNIP
  });

  host_accessor A{fpBuf};
  bool passed = true;
  for (std::size_t i = 0; i < size; ++i) {
    const float b = 4.0f * static_cast<float>(i);
    if (1.0 < A[i] - (static_cast<float>(i) +
                      std::sqrt(std::pow(b, 2) + std::pow(b + 1, 2) +
                                std::pow(b + 2, 2) + std::pow(b + 3, 2)))) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
