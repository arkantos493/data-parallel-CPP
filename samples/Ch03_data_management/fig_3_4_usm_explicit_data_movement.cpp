// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <array>
using namespace sycl;

int main() {
  constexpr std::size_t N = 42;
  std::array<int, N> host_array{};
  host_array.fill(N);

  queue Q{};

  int* device_array = malloc_device<int>(N, Q);

  // We will learn how to simplify this example later
  Q.submit([&](handler& h) {
    // copy hostArray to deviceArray
    h.memcpy(device_array, &host_array[0], N * sizeof(int));
  });
  Q.wait();

  Q.submit([&](handler& h) {
    h.parallel_for(N, [=](id<1> i) { device_array[i]++; });
  });
  Q.wait();

  Q.submit([&](handler& h) {
    // copy deviceArray back to hostArray
    h.memcpy(&host_array[0], device_array, N * sizeof(int));
  });
  Q.wait();

  free(device_array, Q);

  // Check that all outputs match expected value
  const bool passed =
      std::all_of(host_array.begin(), host_array.end(),
                  [](const int i) { return i == static_cast<int>(N + 1); });
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
