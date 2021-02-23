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

  // Nuance: Create new scope so that we can easily cause buffers to go out
  // of scope and be destroyed
  {
    // Create buffers using host allocations (vector in this case)
    buffer in_buf{in_vec};
    buffer out_buf{out_vec};

    // Submit the kernel to the queue
    Q.submit([&](handler& h) {
      accessor in{in_buf, h};
      accessor out{out_buf, h};

      h.parallel_for(range{N}, [=](id<1> idx) { out[idx] = in[idx]; });
    });

    // Close the scope that buffer is alive within!  Causes buffer destruction
    // which will wait until the kernels writing to buffers have completed, and
    // will copy the data from written buffers back to host allocations (our
    // std::vectors in this case).  After the buffer destructor runs, caused by
    // this closing of scope, then it is safe to access the original in_vec and
    // out_vec again!
  }

  // Check that all outputs match expected value
  // WARNING: The buffer destructor must have run for us to safely use in_vec
  // and out_vec again in our host code.  While the buffer is alive it owns
  // those allocations, and they are not safe for us to use!  At the least they
  // will contain values that are not up to date.  This code is safe and correct
  // because the closing of scope above has caused the buffer to be destroyed
  // before this point where we use the vectors again.
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
