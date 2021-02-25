// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

// https://community.intel.com/t5/Intel-oneAPI-Threading-Building/tbb-task-has-not-been-declared/m-p/1255723?profile.language=de&countrylabel=Mexico
#if defined(__GNUG__)
#define PSTL_USE_PARALLEL_POLICIES 0
#define _GLIBCXX_USE_TBB_PAR_BACKEND 0
#endif

#include <CL/sycl.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

using namespace sycl;
#if defined(__GNUG__)
namespace dpstd = dpl;
#endif

int main() {
  queue Q{};
  buffer<int> buf{1000};

  auto buf_begin = dpstd::begin(buf);
  auto buf_end = dpstd::end(buf);

  auto policy = dpstd::execution::make_device_policy<class fill>(Q);
  std::fill(policy, buf_begin, buf_end, 42);

  return 0;
}
