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
  constexpr std::size_t size = 1000;

  buffer<int> buf{size};

  auto buf_begin = dpstd::begin(buf);
  auto buf_end = dpstd::end(buf);

  auto policy = dpstd::execution::make_device_policy<class fill>(Q);
  std::fill(policy, buf_begin, buf_end, 42);

  host_accessor h_acc{buf};
  bool passed = true;
  for (std::size_t i = 0; i < size; ++i) {
    if (h_acc[i] != 42) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';
  return passed ? 0 : 1;
}
