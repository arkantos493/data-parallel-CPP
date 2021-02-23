// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
using namespace sycl;

int main() {
  queue Q{};
  buffer<int> buf{1000};

  auto buf_begin = dpstd::begin(buf);
  auto buf_end = dpstd::end(buf);

  auto policy = dpstd::execution::make_device_policy<class fill>(Q);
  std::fill(policy, buf_begin, buf_end, 42);

  return 0;
}
