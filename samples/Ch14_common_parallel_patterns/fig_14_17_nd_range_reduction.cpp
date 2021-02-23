// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>

using namespace sycl;
using namespace sycl::ONEAPI;

int main() {
  using memory_order = sycl::ONEAPI::memory_order;
  using memory_scope = sycl::ONEAPI::memory_scope;

  constexpr std::size_t N = 16;
  constexpr std::size_t B = 4;

  queue Q{};
  int* data = malloc_shared<int>(N, Q);
  std::iota(data, data + N, 1);
  int* sum = malloc_shared<int>(1, Q);
  *sum = 0;

  Q.parallel_for(nd_range<1>{N, B}, [=](nd_item<1> it) {
     const std::size_t i = it.get_global_id(0);
     const int group_sum = reduce(it.get_group(), data[i], plus<>());
     if (it.get_local_id(0) == 0) {
       atomic_ref<int, memory_order::relaxed, memory_scope::system,
                  access::address_space::global_space>(*sum) += group_sum;
     }
   }).wait();

  std::cout << "sum = " << *sum << '\n';
  const bool passed = *sum == ((N * (N + 1)) / 2);
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';

  free(sum, Q);
  free(data, Q);
  return passed ? 0 : 1;
}
