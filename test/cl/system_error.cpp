/*
 * Copyright (c) 2019 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#include <catch2/catch.hpp>
#include <CL/opencl.h>

#include "accxx/cl/system_error.hpp"

TEST_CASE("OpenCL error creation", "[error-code-cl-create]")
{
  std::error_code code = accxx::cl_error_code(CL_SUCCESS);
  REQUIRE(code == accxx::cl_error_code(CL_SUCCESS));
  REQUIRE(code.value() == CL_SUCCESS);
}

TEST_CASE("OpenCL error category", "[error-category-cl]")
{
  std::error_code code = accxx::cl_error_code(CL_SUCCESS);
  REQUIRE(std::string(code.category().name()) == "clError");
}
