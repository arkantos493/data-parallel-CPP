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
    buffer dataBuf{data};

    queue Q{host_selector()};
    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>() << '\n';

    Q.submit([&](handler& h) {
      // BEGIN CODE SNIP
      // This is a typical global accessor.
      accessor dataAcc{dataBuf, h};

      // This is a 1D local accessor consisting of 16 ints:
      accessor<int, 1, access::mode::read_write, access::target::local>
          localIntAcc{16, h};

      // This is a 2D local accessor consisting of 4 x 4 floats:
      accessor<float, 2, access::mode::read_write, access::target::local>
          localFloatAcc{{4, 4}, h};

      h.parallel_for(nd_range<1>{range{size}, range{16}}, [=](nd_item<1> item) {
        const std::size_t index = item.get_global_id();
        const std::size_t local_index = item.get_local_id();

        // Within a kernel, a local accessor may be read from
        // and written to like any other accessor.
        localIntAcc[local_index] = dataAcc[index] + 1;
        dataAcc[index] = localIntAcc[local_index];
      });
      // END CODE SNIP
    });
  }

  // Check that all outputs match serial execution.
  bool passed = true;
  for (std::size_t i = 0; i < size; i++) {
    if (data[i] != static_cast<int>(i + 1)) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
