// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

using namespace sycl;

int main() {
  const std::size_t N = 16;
  const std::size_t M = 16;

  queue Q{};

  range<2> stencil_range{N, M};
  range<2> alloc_range{N + 2, M + 2};
  std::vector<float> input(alloc_range.size());
  std::iota(input.begin(), input.end(), 1);
  std::vector<float> output(alloc_range.size(), 0);

  {
    buffer input_buf{input.data(), alloc_range};
    buffer output_buf{output.data(), alloc_range};

    Q.submit([&](handler& h) {
      accessor input{input_buf, h};
      accessor output{output_buf, h};

      // Compute the average of each cell and its immediate neighbors
      h.parallel_for(stencil_range, [=](id<2> idx) {
        const std::size_t i = idx[0] + 1;
        const std::size_t j = idx[1] + 1;

        const float self = input[i][j];
        const float north = input[i - 1][j];
        const float east = input[i][j + 1];
        const float south = input[i + 1][j];
        const float west = input[i][j - 1];
        output[i][j] = (self + north + east + south + west) / 5.0f;
      });
    });
  }

  // Check that all outputs match serial execution.
  bool passed = true;
  for (std::size_t i = 1; i < N + 1; ++i) {
    for (std::size_t j = 1; j < M + 1; ++j) {
      const float self = input[i * (M + 2) + j];
      const float north = input[(i - 1) * (M + 2) + j];
      const float east = input[i * (M + 2) + (j + 1)];
      const float south = input[(i + 1) * (M + 2) + j];
      const float west = input[i * (M + 2) + (j - 1)];
      const float gold = (self + north + east + south + west) / 5.0f;
      if (std::abs(output[i * (M + 2) + j] - gold) >= 1.0E-06) {
        passed = false;
      }
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
