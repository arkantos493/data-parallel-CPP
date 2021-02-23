// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
using namespace sycl;

int main() {
  constexpr std::size_t size = 16;
  std::array<int, size> data{};
  std::iota(data.begin(), data.end(), 0);

  {
    buffer data_buf{data};

    queue Q{};
    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>() << '\n';

    Q.submit([&](handler& h) {
      accessor data_acc{data_buf, h};

      // BEGIN CODE SNIP
      h.parallel_for(nd_range{{size}, {16}}, [=](nd_item<1> item) {
        ONEAPI::sub_group sg = item.get_sub_group();
        const std::size_t index = item.get_global_id();
        sg.barrier();
        data_acc[index] = data_acc[index] + 1;
      });
      // END CODE SNIP
    });
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
