// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;
using namespace sycl::ONEAPI;

int main() {
  constexpr std::size_t N = 32;
  constexpr std::size_t M = 4;
  std::vector<int> data(N, 0);

  queue Q{};
  {
    buffer buf{data};

    Q.submit([&](handler& h) {
      atomic_accessor acc{buf, h, relaxed_order, system_scope};
      h.parallel_for(N, [=](id<1> i) {
        const std::size_t j = i % M;
        acc[j] += 1;
      });
    });
  }

  for (std::size_t i = 0; i < N; ++i) {
    std::cout << "data [" << i << "] = " << data[i] << '\n';
  }

  std::array<int, N> gold{};
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t j = i % M;
    gold[j] += 1;
  }
  bool passed = true;
  for (std::size_t i = 0; i < N; ++i) {
    if (data[i] != gold[i]) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
