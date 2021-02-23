// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;

int main() {
  constexpr std::size_t N = 4;

  queue Q{property::queue::in_order()};

  Q.submit([&](handler& h) {
    h.parallel_for(N, [=]([[maybe_unused]] id<1> i) { /*...*/ });  // Task A
  });
  Q.submit([&](handler& h) {
    h.parallel_for(N, [=]([[maybe_unused]] id<1> i) { /*...*/ });  // Task B
  });
  Q.submit([&](handler& h) {
    h.parallel_for(N, [=]([[maybe_unused]] id<1> i) { /*...*/ });  // Task C
  });

  return 0;
}
