/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#include <catch2/catch.hpp>
#include <cuda_runtime_api.h>

#include "accxx/cuda/call.hpp"
#include "accxx/cuda/cuda_device_guard.hpp"

TEST_CASE("CUDA set device", "[cuda-device-guard-success]")
{
  int devices{-1};
  ACCXX_CUDA_CALL(cudaGetDeviceCount(&devices));

  for (int i = 0; i < devices; ++i)
  {
    accxx::cuda_device_guard g{i};

    int device_id{-1};
    ACCXX_CUDA_CALL(cudaGetDevice(&device_id));
    REQUIRE(i == device_id);
  }
}

