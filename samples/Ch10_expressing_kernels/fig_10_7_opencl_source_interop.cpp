// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <numeric>
using namespace sycl;

int main() {
  constexpr std::size_t size = 16;
  std::array<int, size> data{};
  std::iota(data.begin(), data.end(), 0);

  {
    buffer data_buf{data};

    // BEGIN CODE SNIP
    // Note: This must select a device that supports interop!
    queue Q{cpu_selector{}};

    program p{Q.get_context()};
    p.build_with_source(R"CLC(
            kernel void add(global int* data) {
                const int index = get_global_id(0);
                data[index] = data[index] + 1;
            }
        )CLC",
                        "-cl-fast-relaxed-math");

    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>() << '\n';

    Q.submit([&](handler& h) {
      accessor data_acc{data_buf, h};

      h.set_args(data_acc);
      h.parallel_for(size, p.get_kernel("add"));
    });
    // END CODE SNIP
  }

  // Check that all outputs match serial execution.
  bool passed = true;
  for (std::size_t i = 0; i < size; ++i) {
    if (data[i] != static_cast<int>(i + 1)) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
