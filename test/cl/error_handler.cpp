/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#include <catch2/catch.hpp>
#include <CL/cl.h>

#include "accxx/cl/error_handler.hpp"
#include "accxx/cl/system_error.hpp"

TEST_CASE("OpenCL change error handler", "[cl-error-handler-set]")
{
  int i{};
  auto old_handler = accxx::set_cl_error_handler([&i](std::error_code ec) {
    REQUIRE(ec.value() == CL_SUCCESS);
    ++i;
  });
  accxx::get_cl_error_handler()(make_error_code(accxx::cl_error_code(CL_SUCCESS)));
  REQUIRE(i == 1);
  accxx::set_cl_error_handler(old_handler);
}

TEST_CASE("OpenCL change noexcept error handler", "[cl-noexcept-error-handler-set]")
{
  int i{};
  auto old_handler = accxx::set_noexcept_cl_error_handler([&i](std::error_code ec) {
    REQUIRE(ec.value() == CL_SUCCESS);
    ++i;
  });
  accxx::get_noexcept_cl_error_handler()(make_error_code(accxx::cl_error_code(CL_SUCCESS)));
  REQUIRE(i == 1);
  accxx::set_noexcept_cl_error_handler(old_handler);
}

