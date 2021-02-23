// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
using namespace sycl;

int main() {
  constexpr size_t N = 42;
  std::array<int, N> my_data{};
  my_data.fill(0);

  {
    queue q{};
    buffer my_buffer{my_data};

    q.submit([&](handler& h) {
      // create an accessor to update
      // the buffer on the device
      accessor my_accessor{my_buffer, h};

      h.parallel_for(N, [=](id<1> i) { my_accessor[i]++; });
    });

    // create host accessor
    host_accessor host_accessor{my_buffer};

    // for (const int val : host_accessor) {
    //   std::cout << val << ' ';
    // }
    for (std::size_t i = 0; i < N; ++i) {
      // access myBuffer on host
      std::cout << host_accessor[i] << ' ';
    }
    std::cout << '\n';
  }

  // myData is updated when myBuffer is
  // destroyed upon exiting scope
  for (const int val : my_data) {
    std::cout << val << ' ';
  }
  std::cout << '\n';
}
