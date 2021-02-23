// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>  // For fpga_selector
#include <iostream>
#include <string_view>
using namespace sycl;

void select_device(const device_selector& selector,
                   const std::string_view selector_name) {
  std::cout << selector_name << ": ";
  try {
    const device dev{selector};
    std::cout << "Selected device: " << dev.get_info<info::device::name>()
              << '\n';
    std::cout << "                  -> Device vendor: "
              << dev.get_info<info::device::vendor>() << '\n';
  } catch (exception& e) {
    std::cout << "No suitable device found!\n";
  }
}

int main() {
  select_device(default_selector{}, "default_selector");
  select_device(host_selector{}, "host_selector");
  select_device(cpu_selector{}, "cpu_selector");
  select_device(gpu_selector{}, "gpu_selector");
  select_device(accelerator_selector{}, "accelerator_selector");
  select_device(INTEL::fpga_selector{}, "fpga_selector");

  return 0;
}
