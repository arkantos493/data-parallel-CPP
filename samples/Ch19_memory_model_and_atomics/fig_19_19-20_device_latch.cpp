// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

using namespace sycl;
using namespace sycl::ONEAPI;

class device_latch {
  using memory_order = ONEAPI::memory_order;
  using memory_scope = ONEAPI::memory_scope;

 public:
  explicit device_latch(std::size_t num_groups)
      : counter{0}, expected{num_groups} {}

  template <int Dimensions>
  void arrive_and_wait(nd_item<Dimensions>& it) {
    it.barrier();
    // Elect one work-item per work-group to be involved in the synchronization
    // All other work-items wait at the barrier after the branch
    if (it.get_local_linear_id() == 0) {
      atomic_ref<size_t, memory_order::acq_rel, memory_scope::device,
                 access::address_space::global_space>
          atomic_counter{counter};

      // Signal arrival at the barrier
      // Previous writes should be visible to all work-items on the device
      atomic_counter++;

      // Wait for all work-groups to arrive
      // Synchronize with previous releases by all work-items on the device
      while (atomic_counter.load() != expected) {
      }
    }
    it.barrier();
  }

 private:
  std::size_t counter;
  std::size_t expected;
};

int main() {
  queue Q{};

  // The number of groups here must be chosen carefully to guarantee forward
  // progress!
  constexpr std::size_t num_groups = 8;
  constexpr std::size_t items_per_group = 64;
  nd_range<1> R{num_groups * items_per_group, items_per_group};

  // Allocate two arrays in USM
  // The first will be used for communication
  // The second will be used for validation
  std::size_t* data =
      sycl::malloc_shared<std::size_t>(num_groups * items_per_group, Q);
  std::size_t* sums =
      sycl::malloc_shared<std::size_t>(num_groups * items_per_group, Q);

  // Allocate a one-time-use device_latch in USM
  void* ptr = sycl::malloc_shared(sizeof(device_latch), Q);
  device_latch* latch = new (ptr) device_latch{num_groups};
  Q.submit([&](handler& h) {
     h.parallel_for(R, [=](nd_item<1> it) {
       // Every work-item writes a 1 to its location
       data[it.get_global_linear_id()] = 1;

       // Every work-item waits for all writes
       latch->arrive_and_wait(it);

       // Every work-item sums the values it can see
       std::size_t sum = 0;
       for (std::size_t i = 0; i < num_groups * items_per_group; ++i) {
         sum += data[i];
       }
       sums[it.get_global_linear_id()] = sum;
     });
   }).wait();
  free(ptr, Q);

  // Check that all work-items saw all writes
  bool passed = true;
  for (std::size_t i = 0; i < num_groups * items_per_group; ++i) {
    if (sums[i] != num_groups * items_per_group) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';

  free(sums, Q);
  free(data, Q);
  return passed ? 0 : 1;
}
