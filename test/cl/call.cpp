/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#include <CL/cl.h>
#include <catch2/catch.hpp>
#include <stdexcept>

#include "accxx/cl/call.hpp"

namespace {

struct test_cl_call_fixture
{
  accxx::error_handler m_old_handler{};
  bool failed{false};

  test_cl_call_fixture() noexcept :
    m_old_handler{accxx::set_cl_error_handler([&](std::error_code code) {
      REQUIRE(code.value() != CL_SUCCESS);
      failed = true;
    })}
  { }

  test_cl_call_fixture(test_cl_call_fixture const&) = delete;
  test_cl_call_fixture(test_cl_call_fixture&&)      = delete;

  ~test_cl_call_fixture()
  {
    accxx::set_cl_error_handler(m_old_handler);
  }

  test_cl_call_fixture& operator=(test_cl_call_fixture const&) = delete;
  test_cl_call_fixture& operator=(test_cl_call_fixture&&) = delete;
};

} // namespace

TEST_CASE_METHOD(test_cl_call_fixture, "OpenCL call success", "[cl-call-success]")
{
  cl_uint num_platforms{};
  ACCXX_CL_CALL(clGetPlatformIDs(0, nullptr, &num_platforms));
  REQUIRE(failed == false);
  REQUIRE(num_platforms > 0);
}

TEST_CASE_METHOD(test_cl_call_fixture, "OpenCL call fail", "[cl-call-fail]")
{
  ACCXX_CL_CALL(clGetPlatformIDs(0, nullptr, nullptr));
  REQUIRE(failed == true);
}
