// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

using namespace sycl;
using namespace sycl::ONEAPI;

template <typename T, int dimensions>
using local_accessor =
    accessor<T, dimensions, access::mode::read_write, access::target::local>;

int main() {
  // Set parameters to control neighborhood size
  constexpr float CUTOFF = 3.0f;
  constexpr unsigned int MAX_K = 150;

  queue Q{};

  // Initialize input and output on the host
  constexpr std::size_t Nx = 8;
  constexpr std::size_t Ny = 8;
  constexpr std::size_t Nz = 8;
  constexpr std::size_t N = Nx * Ny * Nz;
  float3* position = malloc_shared<float3>(N, Q);
  for (std::size_t x = 0; x < Nx; ++x) {
    for (std::size_t y = 0; y < Ny; ++y) {
      for (std::size_t z = 0; z < Nz; ++z) {
        position[z * Ny * Nx + y * Nx + x] = {x, y, z};
      }
    }
  }
  unsigned int* num_neighbors = malloc_shared<unsigned int>(N, Q);
  std::fill(num_neighbors, num_neighbors + N, 0);
  unsigned int* neighbors = malloc_shared<unsigned int>(N * MAX_K, Q);
  std::fill(neighbors, neighbors + N * MAX_K, 0);

  range<2> global{N, 8};
  range<2> local{1, 8};
  Q.parallel_for(
       nd_range<2>{global, local},
       [=](nd_item<2> it) [[intel::reqd_sub_group_size(8)]] {
         const std::size_t i = it.get_global_id(0);
         const sub_group sg = it.get_sub_group();
         const id<1> sglid = sg.get_local_id();
         const range<1> sgrange = sg.get_max_local_range();

         unsigned int k = 0;
         for (std::size_t j = sglid; j < N; j += sgrange[0]) {
           // Compute distance between i and neighbor j
           const float r = distance(position[i], position[j]);

           // Pack neighbors that require post-processing into a list
           const unsigned int pack = (i != j) && (r <= CUTOFF);
           const unsigned int offset = exclusive_scan(sg, pack, plus<>());
           if (pack) {
             neighbors[i * MAX_K + k + offset] = j;
           }

           // Keep track of how many neighbors have been packed so far
           k += reduce(sg, pack, plus<>());
         }
         num_neighbors[i] = reduce(sg, k, maximum<>());
       })
      .wait();

  // Check that all outputs match serial execution
  bool passed = true;
  for (std::size_t i = 0; i < N; ++i) {
    unsigned int k = 0;
    for (std::size_t j = 0; j < N; ++j) {
      const float r = distance(position[i], position[j]);
      if (i != j and r <= CUTOFF) {
        if (neighbors[i * MAX_K + k] != j) {
          passed = false;
        }
        k++;
      }
    }
    if (num_neighbors[i] != k) {
      passed = false;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  free(neighbors, Q);
  free(num_neighbors, Q);
  free(position, Q);
  return passed ? 0 : 1;
}
