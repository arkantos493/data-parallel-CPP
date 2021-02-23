// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <numeric>
using namespace sycl;

int main() {
  constexpr std::size_t array_size = 16;
  std::array<int, array_size> data{};
  std::iota(data.begin(), data.end(), 0);

  buffer dataBuf{data};

  queue Q{host_selector{}};
  Q.submit([&](handler& h) {
    accessor dataAcc{dataBuf, h};

    h.parallel_for(array_size, [=](id<1> i) {
      const std::size_t condition = i[0] & 1;
      if (static_cast<bool>(condition)) {
        dataAcc[i] = dataAcc[i] * 2;  // odd
      } else {
        dataAcc[i] = dataAcc[i] + 1;  // even
      }
    });
  });

  host_accessor dataAcc{dataBuf};
  bool passed = true;
  for (std::size_t i = 0; i < array_size; ++i) {
    if (data[i] != static_cast<int>(i % 2 == 1 ? i * 2 : i + 1)) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
