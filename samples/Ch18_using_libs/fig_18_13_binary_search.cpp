// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

// https://community.intel.com/t5/Intel-oneAPI-Threading-Building/tbb-task-has-not-been-declared/m-p/1255723?profile.language=de&countrylabel=Mexico
#if defined(__GNUG__)
#define PSTL_USE_PARALLEL_POLICIES 0
#define _GLIBCXX_USE_TBB_PAR_BACKEND 0
#endif

#include <CL/sycl.hpp>
#include <cstdint>
#include <iostream>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

using namespace sycl;
#if defined(__GNUG__)
namespace dpstd = dpl;
#endif

int main() {
  buffer<std::uint64_t> kB{range<1>(10)};
  buffer<std::uint64_t> vB{range<1>(5)};
  buffer<std::uint64_t> rB{range<1>(5)};

  {
    accessor k{kB};
    accessor v{vB};

    // Initialize data, sorted
    k[0] = 0;
    k[1] = 5;
    k[2] = 6;
    k[3] = 6;
    k[4] = 7;
    k[5] = 7;
    k[6] = 8;
    k[7] = 8;
    k[8] = 9;
    k[9] = 9;

    v[0] = 1;
    v[1] = 6;
    v[2] = 3;
    v[3] = 7;
    v[4] = 8;
  }

  // create dpc++ iterators
  auto k_beg = dpstd::begin(kB);
  auto k_end = dpstd::end(kB);
  auto v_beg = dpstd::begin(vB);
  auto v_end = dpstd::end(vB);
  auto r_beg = dpstd::begin(rB);

  // create named policy from existing one
  auto policy = dpstd::execution::make_device_policy<class bSearch>(
      dpstd::execution::default_policy);

  // call algorithm
  dpstd::binary_search(policy, k_beg, k_end, v_beg, v_end, r_beg);

  // check data
  accessor r{rB};
  if ((r[0] == false) && (r[1] == true) && (r[2] == false) && (r[3] == true) &&
      (r[4] == true)) {
    std::cout << "Passed. \nRun on "
              << policy.queue().get_device().get_info<info::device::name>()
              << '\n';
    return 0;
  } else {
    std::cout << "failed: values do not match.\n";
    return 1;
  }
}
