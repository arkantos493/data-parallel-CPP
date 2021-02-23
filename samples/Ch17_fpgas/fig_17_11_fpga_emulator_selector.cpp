// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>  // For fpga_selector
using namespace sycl;

int main() {
  queue Q{INTEL::fpga_emulator_selector{}};
  std::cout << "Device: " << Q.get_device().get_info<info::device::name>()
            << '\n';

  Q.submit([&](handler& h) {
    h.parallel_for(1024, [=]([[maybe_unused]] id<1> idx) {
      // ...
    });
  });

  return 0;
}
