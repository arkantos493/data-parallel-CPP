// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

#define QUERY_INFO(obj, query)                                           \
  do {                                                                   \
    std::cout << "\t" #query " is '" << (obj).template get_info<query>() \
              << "'\n";                                                  \
  } while (0)

#define QUERY_TYPE(dev, type)                                               \
  do {                                                                      \
    std::cout << "\tis_" #type "(): " << ((dev).is_##type() ? "Yes" : "No") \
              << '\n';                                                      \
  } while (0)

int main() {
  // Loop through the available platforms
  for (const platform& this_platform : platform::get_platforms()) {
    std::cout << "Found Platform:\n";
    QUERY_INFO(this_platform, info::platform::name);
    QUERY_INFO(this_platform, info::platform::vendor);
    QUERY_INFO(this_platform, info::platform::version);
    QUERY_INFO(this_platform, info::platform::profile);

    // Loop through the devices available in this plaform
    for (const device& dev : this_platform.get_devices()) {
      std::cout << "  Device: " << dev.get_info<info::device::name>() << "\n";
      QUERY_TYPE(dev, host);         // dev.is_host()
      QUERY_TYPE(dev, cpu);          // dev.is_cpu()
      QUERY_TYPE(dev, gpu);          // dev.is_gpu()
      QUERY_TYPE(dev, accelerator);  // dev.is_accelerator()

      QUERY_INFO(dev, info::device::vendor);
      QUERY_INFO(dev, info::device::driver_version);
      QUERY_INFO(dev, info::device::max_work_item_dimensions);
      QUERY_INFO(dev, info::device::max_work_group_size);
      QUERY_INFO(dev, info::device::mem_base_addr_align);
      QUERY_INFO(dev, info::device::partition_max_sub_devices);

      std::cout << "    Many more queries are available than shown here!\n";
    }
    std::cout << '\n';
  }
  return 0;
}
