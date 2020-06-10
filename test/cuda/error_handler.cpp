/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#include <catch2/catch.hpp>
#include <cuda_runtime_api.h>

#include "accxx/cuda/error_handler.hpp"
#include "accxx/cuda/system_error.hpp"

TEST_CASE("CUDA change error handler", "[cuda-error-handler-set]")
{
  int i{};
  auto old_handler = accxx::set_cuda_error_handler([&i](std::error_code ec) {
    REQUIRE(ec.value() == cudaSuccess);
    ++i;
  });
  accxx::get_cuda_error_handler()(make_error_code(cudaSuccess));
  REQUIRE(i == 1);
  accxx::set_cuda_error_handler(old_handler);
}

TEST_CASE("CUDA change noexcept error handler", "[cuda-noexcept-error-handler-set]")
{
  int i{};
  auto old_handler = accxx::set_noexcept_cuda_error_handler([&i](std::error_code ec) {
    REQUIRE(ec.value() == cudaSuccess);
    ++i;
  });
  accxx::get_noexcept_cuda_error_handler()(make_error_code(cudaSuccess));
  REQUIRE(i == 1);
  accxx::set_noexcept_cuda_error_handler(old_handler);
}
