// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>

using namespace sycl;
using namespace sycl::ONEAPI;

int main() {
  using memory_order = sycl::ONEAPI::memory_order;
  using memory_scope = sycl::ONEAPI::memory_scope;

  constexpr std::size_t N = 32;
  constexpr std::size_t M = 4;

  queue Q{};

  int* data = malloc_shared<int>(N, Q);
  std::fill(data, data + N, 0);

  Q.parallel_for(N, [=](id<1> i) {
     const std::size_t j = i % M;
     atomic_ref<int, memory_order::relaxed, memory_scope::system,
                access::address_space::global_space>
         atomic_data{data[j]};
     atomic_data += 1;
   }).wait();

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
  free(data, Q);
  return passed ? 0 : 1;
}
