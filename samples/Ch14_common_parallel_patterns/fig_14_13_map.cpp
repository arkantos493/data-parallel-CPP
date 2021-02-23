// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

using namespace sycl;

int main() {
  constexpr std::size_t N = 64;

  queue Q{};

  float* input = malloc_shared<float>(N, Q);
  std::iota(input, input + N, 1);
  float* output = malloc_shared<float>(N, Q);
  std::fill(output, output + N, 0);

  // Compute the square root of each input value
  Q.parallel_for(N, [=](id<1> i) { output[i] = sqrt(input[i]); }).wait();

  // Check that all outputs match serial execution.
  bool passed = true;
  for (std::size_t i = 0; i < N; ++i) {
    float gold = std::sqrt(input[i]);
    if (std::abs(output[i] - gold) >= 1.0E-06) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';

  free(output, Q);
  free(input, Q);
  return passed ? 0 : 1;
}
