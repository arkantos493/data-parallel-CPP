// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
using namespace sycl;

int main() {
  constexpr int n = 10;

  queue Q{};
  usm_allocator<int, usm::alloc::shared> alloc(Q.get_context(), Q.get_device());
  std::vector<int, decltype(alloc)> vec(n, alloc);

  std::fill(dpstd::execution::make_device_policy(Q), vec.begin(), vec.end(),
            78);
  Q.wait();

  return 0;
}
