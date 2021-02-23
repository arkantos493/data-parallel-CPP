// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
using namespace sycl;

constexpr std::size_t num_runs = 10;
constexpr std::size_t scalar = 3;

double triad(const std::vector<double>& vecA, const std::vector<double>& vecB,
             std::vector<double>& vecC) {
  assert(vecA.size() == vecB.size() && vecB.size() == vecC.size());
  const std::size_t array_size = vecA.size();

  double best_time = std::numeric_limits<double>::max();

  queue Q{property::queue::enable_profiling{}};
  std::cout << "Running on device: "
            << Q.get_device().get_info<info::device::name>() << '\n';

  buffer bufA{vecA};
  buffer bufB{vecB};
  buffer bufC{vecC};

  for (std::size_t i = 0; i < num_runs; ++i) {
    event Q_event = Q.submit([&](handler& h) {
      accessor A{bufA, h};
      accessor B{bufB, h};
      accessor C{bufC, h};

      h.parallel_for(array_size,
                     [=](id<1> idx) { C[idx] = A[idx] + B[idx] * scalar; });
    });

    const double exec_time_ns =
        Q_event.get_profiling_info<info::event_profiling::command_end>() -
        Q_event.get_profiling_info<info::event_profiling::command_start>();

    std::cout << "Execution time (iteration " << i
              << ") [sec]: " << (exec_time_ns * 1.0E-9) << '\n';
    best_time = std::min(best_time, exec_time_ns);
  }

  return best_time;
}

int main(int argc, char* argv[]) {
  std::size_t array_size;
  if (argc > 1) {
    array_size = std::stoul(argv[1]);
  } else {
    std::cout << "Run as ./<progname> <arraysize in elements>\n";
    return 1;
  }

  std::cout << "Running with stream size of " << array_size << " elements ("
            << static_cast<double>(array_size * sizeof(double)) / 1024 / 1024
            << "MB)\n";

  std::vector<double> D(array_size, 1.0);
  std::vector<double> E(array_size, 2.0);
  std::vector<double> F(array_size, 0.0);

  double min_time = triad(D, E, F);

  // Check correctness
  for (std::size_t i = 0; i < array_size; ++i) {
    if (F[i] != D[i] + scalar * E[i]) {
      std::cout << "\nResult incorrect (element " << i << " is " << F[i]
                << ")!\n";
      return 1;
    }
  }
  std::cout << "Correct results\n\n";

  const std::size_t triad_bytes = 3 * sizeof(double) * array_size;
  std::cout << "Triad Bytes: " << triad_bytes << '\n';
  std::cout << "Time in sec (fastest run): " << min_time * 1.0E-9 << '\n';

  const double triad_bandwidth =
      1.0E-09 * static_cast<double>(triad_bytes) / (min_time * 1.0E-9);
  std::cout << "Bandwidth of fastest run in GB/s: " << triad_bandwidth << '\n';

  return 0;
}
