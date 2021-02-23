// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <exception>
#include <iostream>

class something_went_wrong : std::exception {};

int main() {
  std::cout << "Hello\n";

  throw(something_went_wrong{});
}
