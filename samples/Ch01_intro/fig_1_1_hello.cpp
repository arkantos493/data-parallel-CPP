// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <cstring>
#include <iostream>
#include <string_view>
using namespace sycl;

int main() {
  const std::string_view secret{
      "Ifmmp-!xpsme\"\012#J(n!tpssz-!Ebwf/!"
      "J(n!bgsbje!J!dbo(u!ep!uibu/#!.!IBM"};
  const std::size_t sz = secret.size();

  queue Q{};

  char* result = malloc_shared<char>(sz, Q);
  std::memcpy(result, secret.data(), sz);

  Q.parallel_for(sz, [=](id<1> i) { result[i] -= 1; }).wait();

  std::cout << result << '\n';
  return 0;
}
