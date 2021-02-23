// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
using namespace sycl;

int main() {
  // BEGIN CODE SNIP

  constexpr std::size_t N = 1024;

  // Set up queue on any available device
  queue Q{};

  // Create buffers of size N
  buffer<int> in_buf{N};
  buffer<int> out_buf{N};

  // Use host accessors to initialize the data
  {  // CRITICAL: Begin scope for host_accessor lifetime!
    host_accessor in_acc{in_buf};
    host_accessor out_acc{out_buf};
    for (std::size_t i = 0; i < N; ++i) {
      in_acc[i] = i;
      out_acc[i] = 0;
    }
  }  // CRITICAL: Close scope to make host accessors go out of scope!

  // Submit the kernel to the queue
  Q.submit([&](handler& h) {
    accessor in{in_buf, h};
    accessor out{out_buf, h};

    h.parallel_for(range{N}, [=](id<1> idx) { out[idx] = in[idx]; });
  });

  // Check that all outputs match expected value
  // Use host accessor!  Buffer is still in scope / alive
  host_accessor A{out_buf};
  bool passed = true;
  for (std::size_t i = 0; i < N; ++i) {
    std::cout << "A[" << i << "] = " << A[i] << '\n';
    if (A[i] != static_cast<int>(i)) {
      passed = false;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';

  // END CODE SNIP

  return passed ? 0 : 1;
}
