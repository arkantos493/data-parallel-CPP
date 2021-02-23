// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  queue Q{};
  device dev = Q.get_device();

  std::cout << "We are running on: " << dev.get_info<info::device::name>()
            << '\n';

  // Query results like the following can be used to calculate how
  // large your kernel invocations should be.
  const std::size_t maxWG = dev.get_info<info::device::max_work_group_size>();
  const std::size_t maxGmem = dev.get_info<info::device::global_mem_size>();
  const std::size_t maxLmem = dev.get_info<info::device::local_mem_size>();

  std::cout << "Max WG size is " << maxWG << '\n'
            << "Max Global memory size is " << maxGmem << '\n'
            << "Max Local memory size is " << maxLmem << '\n';

  return 0;
}
