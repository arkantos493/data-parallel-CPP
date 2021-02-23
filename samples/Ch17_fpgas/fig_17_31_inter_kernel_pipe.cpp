// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>  // For fpga_selector
#include <array>
#include <numeric>
using namespace sycl;

int main() {
  constexpr std::size_t count = 1024;
  std::array<int, count> in_array{};
  std::iota(in_array.begin(), in_array.end(), 0);

  // Buffer initialized from in_array (std::array)
  buffer B_in{in_array};

  // Uninitialized buffer with count elements
  buffer<int> B_out{range{count}};

  // Acquire queue to emulated FPGA device
  queue Q{INTEL::fpga_emulator_selector{}};

  // BEGIN CODE SNIP
  // Create alias for pipe type so that consistent across uses
  using my_pipe = pipe<class some_pipe, int>;

  // ND-range kernel
  Q.submit([&](handler& h) {
    accessor A{B_in, h};

    h.parallel_for(count, [=](id<1> idx) { my_pipe::write(A[idx]); });
  });

  // Single_task kernel
  Q.submit([&](handler& h) {
    accessor A{B_out, h};

    h.single_task([=]() {
      for (std::size_t i = 0; i < count; ++i) {
        A[i] = my_pipe::read();
      }
    });
  });

  // END CODE SNIP

  host_accessor A{B_out};
  bool passed = true;
  for (std::size_t i = 0; i < count; ++i) {
    if (A[i] != static_cast<int>(i)) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
