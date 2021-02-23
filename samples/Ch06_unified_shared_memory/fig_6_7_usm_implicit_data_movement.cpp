// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;

int main() {
  constexpr std::size_t N = 42;

  queue Q{};

  int* host_array = malloc_host<int>(N, Q);
  std::iota(host_array, host_array + N, 0);
  int* shared_array = malloc_shared<int>(N, Q);

  Q.submit([&](handler& h) {
    h.parallel_for(N, [=](id<1> i) {
      // access sharedArray and hostArray on device
      shared_array[i] = host_array[i] + 1;
    });
  });
  Q.wait();

  // Check that all outputs match expected value
  bool passed = true;
  for (std::size_t i = 0; i < N; ++i) {
    if (shared_array[i] != static_cast<int>(i + 1)) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';

  free(shared_array, Q);
  free(host_array, Q);

  return passed ? 0 : 1;
}
