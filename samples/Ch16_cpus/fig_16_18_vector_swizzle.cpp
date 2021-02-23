// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;
using namespace sycl::INTEL;

int main() {
  queue Q{};

  bool* passed = malloc_shared<bool>(1, Q);
  *passed = true;

  Q.single_task([=]() {
     sycl::vec<int, 4> old_v = {0, 100, 200, 300};
     sycl::vec<int, 4> new_v{};

     new_v.rgba() = old_v.abgr();
     int vals[] = {300, 200, 100, 0};

     if (new_v.r() != vals[0] || new_v.g() != vals[1] || new_v.b() != vals[2] ||
         new_v.a() != vals[3]) {
       *passed = false;
     }
   }).wait();

  std::cout << (*passed ? "Correct results" : "Wrong results") << '\n';
  const bool p = *passed;
  free(passed, Q);
  return p ? 0 : 1;
}
