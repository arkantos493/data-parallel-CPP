// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>

using namespace sycl;

template <typename T, int dimensions>
using local_accessor =
    accessor<T, dimensions, access::mode::read_write, access::target::local>;

int main() {
  constexpr std::size_t N = 16;
  constexpr std::size_t M = 16;

  queue Q{};

  range<2> stencil_range{N, M};
  range<2> alloc_range{N + 2, M + 2};
  std::vector<float> input(alloc_range.size());
  std::iota(input.begin(), input.end(), 1);
  std::vector<float> output(alloc_range.size(), 0);

  {
    // Create SYCL buffers associated with input/output
    buffer<float, 2> input_buf(input.data(), alloc_range);
    buffer<float, 2> output_buf(output.data(), alloc_range);

    Q.submit([&](handler& h) {
      accessor input{input_buf, h};
      accessor output{output_buf, h};

      constexpr std::size_t B = 4;
      range<2> local_range{B, B};
      range<2> tile_size =
          local_range + range<2>{2, 2};  // Includes boundary cells
      local_accessor<float, 2> tile{tile_size, h};

      // Compute the average of each cell and its immediate neighbors
      h.parallel_for(
          nd_range<2>(stencil_range, local_range), [=](nd_item<2> it) {
            // Load this tile into work-group local memory
            const id<2> lid = it.get_local_id();
            const range<2> lrange = it.get_local_range();
            for (std::size_t ti = lid[0]; ti < B + 2; ti += lrange[0]) {
              const std::size_t gi = ti + B * it.get_group(0);
              for (std::size_t tj = lid[1]; tj < B + 2; tj += lrange[1]) {
                const std::size_t gj = tj + B * it.get_group(1);
                tile[ti][tj] = input[gi][gj];
              }
            }
            it.barrier(access::fence_space::local_space);

            // Compute the stencil using values from local memory
            const std::size_t gi = it.get_global_id(0) + 1;
            const std::size_t gj = it.get_global_id(1) + 1;

            const std::size_t ti = it.get_local_id(0) + 1;
            const std::size_t tj = it.get_local_id(1) + 1;

            const float self = tile[ti][tj];
            const float north = tile[ti - 1][tj];
            const float east = tile[ti][tj + 1];
            const float south = tile[ti + 1][tj];
            const float west = tile[ti][tj - 1];
            output[gi][gj] = (self + north + east + south + west) / 5.0f;
          });
    });
  }

  // Check that all outputs match serial execution
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
