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

    queue Q{};
    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>() << '\n';

    Q.submit([&](handler& h) {
      // This is a typical global accessor.
      accessor data_acc{data_buf, h};

      // BEGIN CODE SNIP
      range group_size{16};
      range num_groups = size / group_size;

      h.parallel_for_work_group(num_groups, group_size, [=](group<1> group) {
        // This variable is declared at work-group scope, so
        // it is allocated in local memory and accessible to
        // all work-items.
        int localIntArr[16];

        // There is an implicit barrier between code and variables
        // declared at work-group scope and the code and variables
        // at work-item scope.

        group.parallel_for_work_item([&](h_item<1> item) {
          const std::size_t index = item.get_global_id();
          const std::size_t local_index = item.get_local_id();

          // The code at work-item scope can read and write the
          // variables declared at work-group scope.
          localIntArr[local_index] = static_cast<int>(index + 1);
          data_acc[index] = localIntArr[local_index];
        });
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
