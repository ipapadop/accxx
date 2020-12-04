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

namespace {

struct test_cuda_call_fixture
{
  accxx::error_handler m_old_handler{};
  bool failed{false};

  test_cuda_call_fixture() noexcept :
    m_old_handler{accxx::set_cuda_error_handler([&](std::error_code code) {
      REQUIRE(code.value() != cudaSuccess);
      failed = true;
    })}
  { }

  test_cuda_call_fixture(test_cuda_call_fixture const&) = delete;
  test_cuda_call_fixture(test_cuda_call_fixture&&)      = delete;

  ~test_cuda_call_fixture()
  {
    accxx::set_cuda_error_handler(m_old_handler);
  }

  test_cuda_call_fixture& operator=(test_cuda_call_fixture const&) = delete;
  test_cuda_call_fixture& operator=(test_cuda_call_fixture&&) = delete;
};

__global__
void dummy()
{}

} // namespace

TEST_CASE_METHOD(test_cuda_call_fixture, "CUDA call host function success", "[cuda-call-host-success]")
{
  int device_id{-1};
  ACCXX_CUDA_CALL(cudaGetDevice(&device_id));
  REQUIRE(device_id > -1);
  REQUIRE(failed == false);
  REQUIRE(cudaGetLastError() == cudaSuccess);
}

TEST_CASE_METHOD(test_cuda_call_fixture, "CUDA call host function fail", "[cuda-call-host--fail]")
{
  int device_id{-1};
  ACCXX_CUDA_CALL(cudaGetDevice(&device_id));
  CHECK(device_id > -1);
  REQUIRE(failed == false);
  REQUIRE(cudaGetLastError() == cudaSuccess);

  ACCXX_CUDA_CALL(cudaSetDevice(device_id + 1));
  REQUIRE(failed == true);
  REQUIRE(cudaGetLastError() == cudaSuccess);
}

TEST_CASE_METHOD(test_cuda_call_fixture, "CUDA call kernel success", "[cuda-call-kernel-success]")
{
  ACCXX_CUDA_CALL(dummy<<<1, 1>>>());
  REQUIRE(failed == false);
  REQUIRE(cudaGetLastError() == cudaSuccess);
}

TEST_CASE_METHOD(test_cuda_call_fixture, "CUDA call kernel fail", "[cuda-call-kernel-fail]")
{
  ACCXX_CUDA_CALL(dummy<<<10000, 100000>>>());
  REQUIRE(failed == true);
  REQUIRE(cudaGetLastError() == cudaSuccess);

  failed = false;

  ACCXX_CUDA_CALL(dummy<<<1, 1>>>());
  REQUIRE(failed == false);
  REQUIRE(cudaGetLastError() == cudaSuccess);
}
