// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

using namespace sycl;
using namespace sycl::ONEAPI;

constexpr std::size_t max_iterations = 1024;
constexpr std::size_t Nx = 1024;
constexpr std::size_t Ny = 768;

struct Parameters {
  float xc;
  float yc;
  float zoom;
  float zoom_px;
  float x_span;
  float y_span;
  float x0;
  float y0;
  float dx;
  float dy;
};

void reset(Parameters params, unsigned int i, unsigned int j,
           unsigned int& count, float& cr, float& ci, float& zr, float& zi) {
  count = 0;
  cr = params.x0 + static_cast<float>(i) * params.dx;
  ci = params.y0 + static_cast<float>(j) * params.dy;
  zr = zi = 0.0f;
}

bool next_iteration(unsigned int i, unsigned int j, unsigned int& count,
                    float& cr, float& ci, float& zr, float& zi,
                    unsigned int* mandelbrot) {
  bool converged = false;
  if (i < Nx) {
    const float next_zr = zr * zr - zi * zi;
    const float next_zi = 2 * zr * zi;
    zr = next_zr + cr;
    zi = next_zi + ci;
    count++;

    // Mark that this value of i has converged
    // Output the i result for this value of i
    if (count >= max_iterations || zr * zr + zi * zi >= 4.0f) {
      converged = true;
      unsigned int px = j * Nx + i;
      mandelbrot[px] = count;
    }
  }
  return converged;
}

int main() {
  queue Q{};

  // Set up parameters to control divergence, image size, etc
  Parameters params{};
  params.xc = 0.0f;
  params.yc = 0.0f;
  params.zoom = 1.0f;
  params.zoom_px = pow(2.0f, 3.0f - params.zoom) * 1e-3f;
  params.x_span = Nx * params.zoom_px;
  params.y_span = Ny * params.zoom_px;
  params.x0 = params.xc - params.x_span * 0.5f;
  params.y0 = params.yc - params.y_span * 0.5f;
  params.dx = params.zoom_px;
  params.dy = params.zoom_px;

  // Initialize output on the host
  unsigned int* mandelbrot = malloc_shared<unsigned int>(Ny * Nx, Q);
  std::fill(mandelbrot, mandelbrot + Ny * Nx, 0);

  range<2> global{Ny, 8};
  range<2> local{1, 8};
  Q.parallel_for(
       nd_range<2>{global, local},
       [=](nd_item<2> it) [[intel::reqd_sub_group_size(8)]] {
         const std::size_t j = it.get_global_id(0);
         const sub_group sg = it.get_sub_group();

         // Treat each row as a queue of i values to compute
         // Initially the head of the queue is at 0
         unsigned int iq = 0;

         // Initially each work-item in the sub-group works on contiguous values
         unsigned int i = iq + sg.get_local_id();
         iq += sg.get_max_local_range()[0];

         // Initialize the iterator variables
         unsigned int count = 0;
         float cr, ci, zr, zi;
         if (i < Nx) {
           reset(params, i, j, count, cr, ci, zr, zi);
         }

         // Keep iterating as long as one work-item has work to do
         while (any_of(sg, i < Nx)) {
           const unsigned int converged =
               next_iteration(i, j, count, cr, ci, zr, zi, mandelbrot);
           if (any_of(sg, converged)) {
             // Replace pixels that have converged using an unpack
             // Pixels that haven't converged are not replaced
             const unsigned int index = exclusive_scan(sg, converged, plus<>());
             i = (converged) ? iq + index : i;
             iq += reduce(sg, converged, plus<>());

             // Reset the iterator variables for the new i
             if (converged) {
               reset(params, i, j, count, cr, ci, zr, zi);
             }
           }
         }
       })
      .wait();

  // Produce an image as a PPM file
  constexpr std::size_t max_color = 65535;
  std::ofstream ppm{};
  ppm.open("mandelbrot.ppm");
  ppm << "P6\n" << Nx << '\n' << Ny << '\n' << max_color << '\n';
  const std::size_t eof = ppm.tellp();
  ppm.close();
  ppm.open("mandelbrot.ppm", std::ofstream::binary | std::ofstream::app);
  ppm.seekp(static_cast<long>(eof));
  std::vector<std::uint16_t> colors(Nx * Ny * 3);
  for (std::size_t px = 0; px < Nx * Ny; ++px) {
    const std::uint16_t color =
        (max_iterations - mandelbrot[px]) *
        (max_color / static_cast<double>(max_iterations));
    colors[3 * px + 0] = color;
    colors[3 * px + 1] = color;
    colors[3 * px + 2] = color;
  }
  ppm.write(reinterpret_cast<char*>(colors.data()), 2 * colors.size());
  ppm.close();

  free(mandelbrot, Q);
  return 0;
}
