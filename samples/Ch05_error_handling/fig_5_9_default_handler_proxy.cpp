// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

// BEGIN CODE SNIP

// Our simple asynchronous handler function
auto handle_async_error = [](exception_list elist) {
  for (auto& e : elist) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception& e) {
      // Print information about the asynchronous exception
    }
  }

  // Terminate abnormally to make clear to user that something unhandled
  // happened
  std::terminate();
};

// END CODE SNIP

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
