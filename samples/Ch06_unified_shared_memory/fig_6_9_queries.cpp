// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;

template <std::size_t N, typename T>
void foo(T data, id<1> i) {
  data[i] = N;
}

int main() {
  constexpr std::size_t N = 42;
  queue Q{};
  device dev = Q.get_device();
  context ctxt = Q.get_context();
  const bool usm_shared = dev.get_info<info::device::usm_shared_allocations>();
  const bool usm_device = dev.get_info<info::device::usm_device_allocations>();
  const bool use_USM = usm_shared || usm_device;

  if (use_USM) {
    int* data;
    if (usm_shared) {
      data = malloc_shared<int>(N, Q);
    } else /* use device allocations */ {
      data = malloc_device<int>(N, Q);
    }
    std::cout << "Using USM with "
              << (get_pointer_type(data, ctxt) == usm::alloc::shared ? "shared"
                                                                     : "device")
              << " allocations on "
              << get_pointer_device(data, ctxt).get_info<info::device::name>()
              << '\n';
    Q.parallel_for(N, [=](id<1> i) { foo<N>(data, i); });
    Q.wait();
    free(data, Q);
  } else /* use buffers */ {
    buffer<int> data{range{N}};
    Q.submit([&](handler& h) {
      accessor a{data, h};
      h.parallel_for(N, [=](id<1> i) { foo<N>(a, i); });
    });
    Q.wait();
  }
  return 0;
}
