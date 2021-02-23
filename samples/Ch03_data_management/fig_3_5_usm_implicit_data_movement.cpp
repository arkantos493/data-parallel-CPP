// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
using namespace sycl;

int main() {
  constexpr std::size_t N = 42;

  queue Q{};
  int* host_array = malloc_host<int>(N, Q);
  // Initialize hostArray on host
  std::iota(host_array, host_array + N, 0);
  int* shared_array = malloc_shared<int>(N, Q);

  // We will learn how to simplify this example later
  Q.submit([&](handler& h) {
    h.parallel_for(N, [=](id<1> i) {
      // access sharedArray and hostArray on device
      shared_array[i] = host_array[i] + 1;
    });
  });
  Q.wait();

  for (std::size_t i = 0; i < N; ++i) {
    // access sharedArray on host
    host_array[i] = shared_array[i];
  }

  free(shared_array, Q);
  free(host_array, Q);

  // Check that all outputs match expected value
  for (std::size_t i = 0; i < N; ++i) {
    if (host_array[i] != static_cast<int>(i + 1)) {
      std::cout << "Wrong results\n";
      return 1;
    }
  }
  std::cout << "Correct results\n";
  return 0;
}
