// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>

using namespace sycl;

int main() {
  constexpr std::size_t N = 32;
  constexpr std::size_t M = 4;

  queue Q{};

  int* data = malloc_shared<int>(N, Q);
  std::fill(data, data + N, 0);

  // Launch exactly one work-group
  // Number of work-groups = global / local
  range<1> global{N};
  range<1> local{N};

  Q.parallel_for(nd_range<1>{global, local}, [=](nd_item<1> it) {
     const std::size_t i = it.get_global_id(0);
     const std::size_t j = i % M;
     for (std::size_t round = 0; round < N; ++round) {
       // Allow exactly one work-item update per round
       if (i == round) {
         data[j] += 1;
       }
       it.barrier();
     }
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
