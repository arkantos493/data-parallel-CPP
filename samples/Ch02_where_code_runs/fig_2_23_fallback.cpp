// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
using namespace sycl;

int main() {
  constexpr std::size_t global_size = 16;
  constexpr std::size_t local_size = 16;
  buffer<std::size_t, 2> B{range{global_size, global_size}};

  queue gpu_Q{gpu_selector{}};
  queue host_Q{host_selector{}};

  nd_range NDR{range{global_size, global_size}, range{local_size, local_size}};

  gpu_Q.submit(
      [&](handler& h) {
        accessor acc{B, h};

        h.parallel_for(NDR, [=](nd_item<2> item) {
          const id<2> ind = item.get_global_id();
          acc[ind] = ind[0] + ind[1];
        });
      },
      host_Q); /** <<== Fallback Queue Specified **/

  host_accessor acc{B};
  for (std::size_t i = 0; i < global_size; ++i) {
    for (std::size_t j = 0; j < global_size; ++j) {
      if (acc[i][j] != i + j) {
        std::cout << "Wrong result\n";
        return 1;
      }
    }
  }
  std::cout << "Correct results\n";
  return 0;
}
