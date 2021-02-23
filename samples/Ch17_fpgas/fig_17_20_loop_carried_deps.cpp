// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>  // For fpga_selector
using namespace sycl;

int generate_random_number([[maybe_unused]] const int state) {
  return 0;  // Useless non-RNG generator as proxy!
};

int main() {
  constexpr std::size_t size = 64;
  queue Q{INTEL::fpga_emulator_selector{}};

  buffer<int> B{range{size}};

  Q.submit([&](handler& h) {
    accessor output{B, h};

    h.single_task([=]() {
      // BEGIN CODE SNIP
      int a = 0;
      for (std::size_t i = 0; i < size; ++i) {
        a = a + static_cast<int>(i);
      }
      // END CODE SNIP
    });
  });

  return 0;
}
