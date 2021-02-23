// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;

int main() {
  constexpr std::size_t N = 4;

  queue Q{};

  event eA = Q.submit([&](handler& h) {
    h.parallel_for(N, [=]([[maybe_unused]] id<1> i) { /*...*/ });  // Task A
  });
  eA.wait();
  event eB = Q.submit([&](handler& h) {
    h.parallel_for(N, [=]([[maybe_unused]] id<1> i) { /*...*/ });  // Task B
  });
  event eC = Q.submit([&](handler& h) {
    h.depends_on(eB);
    h.parallel_for(N, [=]([[maybe_unused]] id<1> i) { /*...*/ });  // Task C
  });
  event eD = Q.submit([&](handler& h) {
    h.depends_on({eB, eC});
    h.parallel_for(N, [=]([[maybe_unused]] id<1> i) { /*...*/ });  // Task D
  });

  return 0;
}
