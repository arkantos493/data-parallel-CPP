// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>  // For fpga_selector
#include <iostream>
using namespace sycl;

int main() {
  queue gpu_queue{gpu_selector{}};
  std::cout << "Selected device 1: "
            << gpu_queue.get_device().get_info<info::device::name>() << '\n';

  queue fpga_queue{INTEL::fpga_selector{}};
  std::cout << "Selected device 2: "
            << fpga_queue.get_device().get_info<info::device::name>() << '\n';

  return 0;
}
