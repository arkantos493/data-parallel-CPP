// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

using namespace sycl;

template <typename T, int dimensions>
using local_accessor =
    accessor<T, dimensions, access::mode::read_write, access::target::local>;

int main() {
  constexpr std::size_t N = 128;
  constexpr std::size_t L = 16;
  constexpr std::size_t G = N / L;

  queue Q{};

  int* input = malloc_shared<int>(N, Q);
  std::iota(input, input + N, 1);
  int* output = malloc_shared<int>(N, Q);
  std::fill(output, output + N, 0);

  // Create a temporary allocation that will only be used by the device
  int* tmp = malloc_device<int>(G, Q);

  // Phase 1: Compute local scans over input blocks
  Q.submit([&](handler& h) {
     local_accessor<int, 1> local{L, h};
     h.parallel_for(nd_range<1>{N, L}, [=](nd_item<1> it) {
       const std::size_t i = it.get_global_id(0);
       const std::size_t li = it.get_local_id(0);

       // Copy input to local memory
       local[li] = input[i];
       it.barrier();

       // Perform inclusive scan in local memory
       for (std::size_t d = 0; d <= log2(static_cast<float>(L)) - 1; ++d) {
         const unsigned int stride = (1 << d);
         const int update = (li >= stride) ? local[li - stride] : 0;
         it.barrier();
         local[li] += update;
         it.barrier();
       }

       // Write the result for each item to the output buffer
       // Write the last result from this block to the temporary buffer
       output[i] = local[li];
       if (li == it.get_local_range()[0] - 1) {
         tmp[it.get_group(0)] = local[li];
       }
     });
   }).wait();

  // Phase 2: Compute scan over partial results
  Q.submit([&](handler& h) {
     local_accessor<int, 1> local{G, h};
     h.parallel_for(nd_range<1>{G, G}, [=](nd_item<1> it) {
       const std::size_t i = it.get_global_id(0);
       const std::size_t li = it.get_local_id(0);

       // Copy input to local memory
       local[li] = tmp[i];
       it.barrier();

       // Perform inclusive scan in local memory
       for (std::size_t d = 0; d <= log2(static_cast<float>(G)) - 1; ++d) {
         const unsigned int stride = (1 << d);
         const int update = (li >= stride) ? local[li - stride] : 0;
         it.barrier();
         local[li] += update;
         it.barrier();
       }

       // Overwrite result from each work-item in the temporary buffer
       tmp[i] = local[li];
     });
   }).wait();

  // Phase 3: Update local scans using partial results
  Q.parallel_for(nd_range<1>{N, L}, [=](nd_item<1> it) {
     const std::size_t g = it.get_group(0);
     if (g > 0) {
       const std::size_t i = it.get_global_id(0);
       output[i] += tmp[g - 1];
     }
   }).wait();

  // Check that all outputs match serial execution
  bool passed = true;
  int gold = 0;
  for (std::size_t i = 0; i < N; ++i) {
    gold += input[i];
    if (output[i] != gold) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';

  free(tmp, Q);
  free(output, Q);
  free(input, Q);
  return passed ? 0 : 1;
}
