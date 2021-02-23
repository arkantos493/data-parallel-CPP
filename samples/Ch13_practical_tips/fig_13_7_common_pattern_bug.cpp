// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
using namespace sycl;

int main() {
  // BEGIN CODE SNIP

  constexpr std::size_t N = 1024;

  // Set up queue on any available device
  queue Q{};

  // Create host containers to initialize on the host
  std::vector<int> in_vec(N);
  std::iota(in_vec.begin(), in_vec.end(), 0);
  std::vector<int> out_vec(N, 0);

  // Create buffers using host allocations (vector in this case)
  buffer in_buf{in_vec};
  buffer out_buf{out_vec};

  // Submit the kernel to the queue
  Q.submit([&](handler& h) {
    accessor in{in_buf, h};
    accessor out{out_buf, h};

    h.parallel_for(range{N}, [=](id<1> idx) { out[idx] = in[idx]; });
  });

  // BUG!!! We're using the host allocation out_vec, but the buffer out_buf
  // is still alive and owns that allocation!  We will probably see the
  // initialiation value (zeros) printed out, since the kernel probably
  // hasn't even run yet, and the buffer has no reason to have copied
  // any output back to the host even if the kernel has run.
  bool passed = true;
  for (std::size_t i = 0; i < N; ++i) {
    std::cout << "out_vec[" << i << "] = " << out_vec[i] << '\n';
    if (out_vec[i] != static_cast<int>(i)) {
      passed = false;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';

  // END CODE SNIP

  return passed ? 0 : 1;
}
