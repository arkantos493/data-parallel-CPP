// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace sycl;

int main() {
  // Initialize input and output memory on the host
  constexpr std::size_t N = 256;

  std::default_random_engine gen(42);
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  auto rng = [&]() { return dist(gen); };

  std::vector<float> a(N * N);
  std::generate(a.begin(), a.end(), rng);
  std::vector<float> b(N * N);
  std::generate(b.begin(), b.end(), rng);
  std::vector<float> c(N * N);

  // Set up queue on any available device
  queue Q{};

  {
    // Create buffers associated with inputs and output
    buffer a_buf{a.data(), range<2>{N, N}};
    buffer b_buf{b.data(), range<2>{N, N}};
    buffer c_buf{c.data(), range<2>{N, N}};

    // Submit the kernel to the queue
    Q.submit([&](handler& h) {
      accessor a{a_buf, h};
      accessor b{b_buf, h};
      accessor c{c_buf, h};

      // START CODE SNIP
      h.parallel_for(range{N, N}, [=](id<2> idx) {
        const std::size_t j = idx[0];
        const std::size_t i = idx[1];
        for (std::size_t k = 0; k < N; ++k) {
          c[j][i] += a[j][k] * b[k][i];  // or c[idx] += a[id(j,k) * b[id(k,i)];
        }
      });
      // END CODE SNIP
    });
  }

  // Check that all outputs match serial execution
  bool passed = true;
  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i < N; ++i) {
      float gold = 0.0;
      for (std::size_t k = 0; k < N; ++k) {
        gold += a[j * N + k] * b[k * N + i];
      }
      if (std::abs(gold - c[j * N + i]) / gold > 1.0E-06) {
        passed = false;
      }
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
