// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
#include <limits>
#include <random>

using namespace sycl;
using namespace sycl::ONEAPI;

template <typename T, typename I>
struct pair {
  bool operator<(const pair& o) const {
    return val <= o.val || (val == o.val && idx <= o.idx);
  }
  T val;
  I idx;
};

template <typename T, typename I>
using minloc = minimum<pair<T, I>>;

int main() {
  constexpr std::size_t N = 16;
  constexpr std::size_t L = 4;

  std::default_random_engine gen(42);
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  auto rng = [&]() { return dist(gen); };

  queue Q{};

  float* data = malloc_shared<float>(N, Q);
  std::generate(data, data + N, rng);
  pair<float, int>* res = malloc_shared<pair<float, int>>(1, Q);

  pair<float, int> identity = {std::numeric_limits<float>::max(),
                               std::numeric_limits<int>::min()};
  *res = identity;

  auto red = reduction(res, identity, minloc<float, int>());

  Q.submit([&](handler& h) {
     h.parallel_for(nd_range<1>{N, L}, red, [=](nd_item<1> item, auto& res) {
       const std::size_t i = item.get_global_id(0);
       pair<float, int> partial = {data[i], static_cast<int>(i)};
       res.combine(partial);
     });
   }).wait();

  std::cout << "minimum value = " << res->val << " at " << res->idx << '\n';

  pair<float, int> gold = identity;
  for (std::size_t i = 0; i < N; ++i) {
    if (data[i] <= gold.val ||
        (data[i] == gold.val && static_cast<int>(i) < gold.idx)) {
      gold.val = data[i];
      gold.idx = static_cast<int>(i);
    }
  }
  const bool passed = (res->val == gold.val) && (res->idx == gold.idx);
  std::cout << (passed ? "Correct results" : "Wrong results") << '\n';

  free(res, Q);
  free(data, Q);
  return passed ? 0 : 1;
}
