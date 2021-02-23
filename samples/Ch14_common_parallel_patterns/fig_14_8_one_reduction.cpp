// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>

using namespace sycl;
using namespace sycl::ONEAPI;

int main() {
  constexpr std::size_t N = 16;
  constexpr std::size_t B = 4;

  queue Q{};
  int* data = malloc_shared<int>(N, Q);
  std::iota(data, data + N, 1);
  int* sum = malloc_shared<int>(1, Q);
  *sum = 0;

  Q.submit([&](handler& h) {
     // BEGIN CODE SNIP
     h.parallel_for(nd_range<1>{N, B}, reduction(sum, plus<>()),
                    [=](nd_item<1> it, auto& sum) {
                      const std::size_t i = it.get_global_id(0);
                      sum += data[i];
                    });
     // END CODE SNIP
   }).wait();

  std::cout << "sum = " << *sum << '\n';
  const bool passed = *sum == ((N * (N + 1)) / 2);
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';

  free(sum, Q);
  free(data, Q);
  return passed ? 0 : 1;
}
