/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#include <catch2/catch.hpp>
#include <cuda_runtime_api.h>
#include <string>

#include "accxx/cuda/system_error.hpp"

TEST_CASE("CUDA error creation", "[cuda-error-code-create]")
{
  std::error_code code = cudaSuccess;
  REQUIRE(code == cudaSuccess);
  REQUIRE(code.value() == cudaSuccess);
}

TEST_CASE("CUDA error category", "[cuda-error-category]")
{
  std::error_code code = cudaSuccess;
  REQUIRE(std::string(code.category().name()) == "cudaError");
}
