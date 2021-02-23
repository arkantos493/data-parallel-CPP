// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <vector>
using namespace sycl;

int main() {
  // Initialize input and output memory on the host
  constexpr std::size_t N = 256;
  std::vector<int> a(N, 1);
  std::vector<int> b(N, 2);
  std::vector<int> c(N);

  // Set up queue on any available device
  queue q{};

  {
    // Create buffers associated with inputs and output
    buffer a_buf{a};
    buffer b_buf{b};
    buffer c_buf{c};

    // Submit the kernel to the queue
    q.submit([&](handler& h) {
      accessor a{a_buf, h};
      accessor b{b_buf, h};
      accessor c{c_buf, h};

      // START CODE SNIP
      h.parallel_for(range{N}, [=](id<1> idx) { c[idx] = a[idx] + b[idx]; });
      // END CODE SNIP
    });
  }

  // Check that all outputs match expected value
  const bool passed =
      std::all_of(c.begin(), c.end(), [](const int i) { return i == 3; });
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
