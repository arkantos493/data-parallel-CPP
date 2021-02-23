// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;

int main() {
  // Appropriate values depend on your HW
  constexpr std::size_t BLOCK_SIZE = 42;
  constexpr std::size_t NUM_BLOCKS = 2;
  constexpr std::size_t N = NUM_BLOCKS * BLOCK_SIZE;

  queue Q{};
  int* data = malloc_shared<int>(N, Q);
  // Never updated after initialization
  int* read_only_data = malloc_shared<int>(BLOCK_SIZE, Q);
  std::iota(read_only_data, read_only_data + N, 0);

  // Mark this data as "read only" so the runtime can copy it
  // to the device instead of migrating it from the host.
  // Real values will be documented by your DPC++ backend.
  int HW_SPECIFIC_ADVICE_RO = 0;
  Q.mem_advise(read_only_data, BLOCK_SIZE,
               static_cast<pi_mem_advice>(HW_SPECIFIC_ADVICE_RO));
  event e = Q.prefetch(data, BLOCK_SIZE);

  for (std::size_t b = 0; b < NUM_BLOCKS; ++b) {
    Q.parallel_for(range{BLOCK_SIZE}, e, [=](id<1> i) {
      data[b * BLOCK_SIZE + i] += read_only_data[i];
    });
    if ((b + 1) < NUM_BLOCKS) {
      // Prefetch next block
      e = Q.prefetch(data + (b + 1) * BLOCK_SIZE, BLOCK_SIZE);
    }
  }
  Q.wait();

  // Check that all outputs match expected value
  bool passed = true;
  for (std::size_t b = 0; b < NUM_BLOCKS; ++b) {
    for (std::size_t i = 0; i < BLOCK_SIZE; ++i) {
      if (data[b * BLOCK_SIZE + i] != static_cast<int>(i)) {
        passed = false;
        break;
      }
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';

  free(data, Q);
  free(read_only_data, Q);
  return passed ? 0 : 1;
}
