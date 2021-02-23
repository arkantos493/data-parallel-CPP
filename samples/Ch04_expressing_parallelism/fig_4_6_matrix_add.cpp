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
  constexpr std::size_t M = 256;
  std::vector<int> a(N * M, 1);
  std::vector<int> b(N * M, 2);
  std::vector<int> c(N * M);

  // Set up queue on any available device
  queue Q{};

  {
    // Create buffers associated with inputs and output
    buffer a_buf{a.data(), range<2>{N, M}};
    buffer b_buf{b.data(), range<2>{N, M}};
    buffer c_buf{c.data(), range<2>{N, M}};

    // Submit the kernel to the queue
    Q.submit([&](handler& h) {
      accessor a{a_buf, h};
      accessor b{b_buf, h};
      accessor c{c_buf, h};

      // START CODE SNIP
      h.parallel_for(range{N, M}, [=](id<2> idx) { c[idx] = a[idx] + b[idx]; });
      // END CODE SNIP
    });
  }

  // Check that all outputs match expected value
  const bool passed =
      std::all_of(c.begin(), c.end(), [](const int i) { return i == 3; });
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
