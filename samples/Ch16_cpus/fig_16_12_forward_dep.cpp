// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;
using namespace sycl::ONEAPI;

int main() {
  constexpr std::size_t n = 16;
  constexpr std::size_t w = 8;

  queue Q{};
  range<2> G = {n, w};
  range<2> L = {1, w};

  int* a = malloc_shared<int>(n * (n + 1), Q);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n + 1; ++j) {
      a[i * n + j] = static_cast<int>(i + j);
    }
  }

  Q.parallel_for(
       nd_range<2>{G, L},
       [=](nd_item<2> it) [[intel::reqd_sub_group_size(w)]] {
         // distribute uniform "i" over the sub-group with 8-way
         // redundant computation
         const std::size_t i = it.get_global_id(0);
         const sub_group sg = it.get_sub_group();

         for (std::size_t j = sg.get_local_id(); j < n; j += w) {
           // load a[i*n+j+1:8] before updating a[i*n+j:8] to preserve
           // loop-carried forward dependency
           const int va = a[i * n + j + 1];
           sg.barrier();
           a[i * n + j] = va + static_cast<int>(i) + 2;
         }
         sg.barrier();
       })
      .wait();

  const bool passed = a[0] == 3 && a[9] == 12;
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  free(a, Q);
  return passed ? 0 : 1;
}
