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

    queue Q{host_selector{}};
    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>() << '\n';

    Q.submit([&](handler& h) {
      accessor data_acc{data_buf, h};

      // In many cases the explicit kernel name template parameter is not
      // required.
      // BEGIN CODE SNIP
      h.parallel_for(size, [=](id<1> i) { data_acc[i] = data_acc[i] + 1; });
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
