// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
using namespace sycl;

int main() {
  constexpr std::size_t workers = 64;
  constexpr std::size_t size = workers * 16;

  // BEGIN CODE SNIP
  std::array<float, size> fpData{};
  fpData.fill(8.0f);

  buffer fpBuf(fpData);

  queue Q{};
  Q.submit([&](handler& h) {
    accessor buf{fpBuf, h};

    h.parallel_for(workers, [=](id<1> idx) {
      float16 inpf16;
      inpf16.load(idx, buf.get_pointer());
      float16 result = inpf16 * 2.0f;
      result.store(idx, buf.get_pointer());
    });
  });
  // END CODE SNIP

  host_accessor hostAcc(fpBuf);
  const bool passed = fpData[0] == 16.0f;
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return (passed ? 0 : 1);
}
