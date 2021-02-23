// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <exception>
using namespace sycl;

// Our simple asynchronous handler function
auto handle_async_error = [](exception_list elist) {
  for (auto& e : elist) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception& e) {
      std::cout << "ASYNC EXCEPTION!!\n";
      std::cout << e.what() << "\n";
    }
  }
};

int main() {
  queue Q1{gpu_selector{}, handle_async_error};
  std::cout << "Device 1: " << Q1.get_device().get_info<info::device::name>()
            << '\n';
  queue Q2{cpu_selector{}, handle_async_error};
  std::cout << "Device 2: " << Q2.get_device().get_info<info::device::name>()
            << '\n';

  try {
    Q1.submit(
        [&]([[maybe_unused]] handler& h) {
          // Empty command group is illegal and generates an error
        },
        Q2);  // Secondary/backup queue!
  } catch (...) {
  }  // Discard regular C++ exceptions for this example
  return 0;
}
