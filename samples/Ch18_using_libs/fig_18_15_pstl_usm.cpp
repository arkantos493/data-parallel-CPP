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

using namespace sycl;
#if defined(__GNUG__)
namespace dpstd = dpl;
#endif

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
