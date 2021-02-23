// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <tuple>

using namespace sycl;
using namespace sycl::ONEAPI;

template <typename T, int dimensions>
using local_accessor =
    accessor<T, dimensions, access::mode::read_write, access::target::local>;

std::tuple<std::size_t, std::size_t> distribute_range(group<1> g,
                                                      std::size_t N) {
  const std::size_t work_per_group = N / g.get_group_range(0);
  const std::size_t remainder = N - g.get_group_range(0) * work_per_group;
  const std::size_t group_start =
      g.get_id(0) * work_per_group + std::min(g.get_id(0), remainder);
  const std::size_t group_end =
      (g.get_id(0) + 1) * work_per_group + std::min(g.get_id(0) + 1, remainder);
  return {group_start, group_end};
}

// Define shorthand aliases for the types of atomic needed by this kernel
namespace {
using memory_order = ONEAPI::memory_order;
using memory_scope = ONEAPI::memory_scope;

template <typename T>
using local_atomic_ref =
    atomic_ref<T, memory_order::relaxed, memory_scope::work_group,
               access::address_space::local_space>;

template <typename T>
using global_atomic_ref =
    atomic_ref<T, memory_order::relaxed, memory_scope::system,
               access::address_space::global_space>;
}  // namespace

int main() {
  constexpr std::size_t num_groups = 72;
  constexpr std::size_t num_items = 16;

  constexpr std::size_t N = 1024;
  constexpr std::size_t B = 64;

  queue Q{};

  std::size_t* input = malloc_shared<std::size_t>(N, Q);
  std::generate(input, input + N, std::mt19937{});
  std::size_t* histogram = malloc_shared<std::size_t>(B, Q);
  std::fill(histogram, histogram + B, 0);

  Q.submit([&](handler& h) {
     local_accessor<std::size_t, 1> local{B, h};
     h.parallel_for(
         nd_range<1>{num_groups * num_items, num_items}, [=](nd_item<1> it) {
           // Phase 1: Work-items co-operate to zero local memory
           for (std::size_t b = it.get_local_id(0); b < B;
                b += it.get_local_range(0)) {
             local[b] = 0;
           }
           it.barrier();  // Wait for all to be zeroed

           // Phase 2: Work-groups each compute a chunk of the input
           // Work-items co-operate to compute histogram in local memory
           group grp = it.get_group();
           const auto [group_start, group_end] = distribute_range(grp, N);
           for (std::size_t i = group_start + it.get_local_id(0); i < group_end;
                i += it.get_local_range(0)) {
             const std::size_t b = input[i] % B;
             local_atomic_ref<std::size_t>(local[b])++;
           }
           it.barrier();  // Wait for all local histogram updates to complete

           // Phase 3: Work-items co-operate to update global memory
           for (std::size_t b = it.get_local_id(0); b < B;
                b += it.get_local_range(0)) {
             global_atomic_ref<std::size_t>(histogram[b]) += local[b];
           }
         });
   }).wait();

  // Compute reference histogram serially on the host
  bool passed = true;
  std::array<std::size_t, B> gold{};
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t b = input[i] % B;
    gold[b]++;
  }
  for (std::size_t b = 0; b < B; ++b) {
    if (gold[b] != histogram[b]) {
      passed = false;
      break;
    }
  }
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';

  free(histogram, Q);
  free(input, Q);
  return passed ? 0 : 1;
}
