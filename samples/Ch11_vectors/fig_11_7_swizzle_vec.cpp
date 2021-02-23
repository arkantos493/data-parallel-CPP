// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
using namespace sycl;

int main() {
  // BEGIN CODE SNIP
  constexpr std::size_t size = 16;

  std::array<float4, size> input{};
  for (std::size_t i = 0; i < size; ++i) {
    input[i] = float4(8.0f, 6.0f, 2.0f, i);
  }

  buffer B{input};

  queue Q{};
  Q.submit([&](handler& h) {
    accessor A{B, h};

    //  We can access the individual elements of a vector by using the
    //  functions x(), y(), z(), w() and so on.
    //
    //  "Swizzles" can be used by calling a vector member equivalent to the
    //  swizzle order that we need, for example zyx() or any combination of the
    //  elements. The swizzle need not be the same size as the original
    //  vector
    h.parallel_for(size, [=](id<1> idx) {
      float4 b = A[idx];
      float w = b.w();
      float4 sw = b.xyzw();
      sw = b.xyzw() * sw.wzyx();
      sw = sw + w;
      A[idx] = sw.xyzw();
    });
  });
  // END CODE SNIP

  host_accessor hostAcc{B};
  bool passed = true;
  for (std::size_t i = 0; i < size; ++i) {
    if (hostAcc[i].y() != 12.0f + i) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
