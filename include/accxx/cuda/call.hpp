/** @file */
/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#ifndef ACCXX_CUDA_CALL_HPP
#define ACCXX_CUDA_CALL_HPP

#include <cuda_runtime_api.h>

#include "accxx/cuda/error_handler.hpp"
#include "accxx/cuda/system_error.hpp"

/// @def ACCXX_CUDA_CALL(...)
/// Calls <tt>...</tt> and checks the return value. If the return value does not indicate success,
/// it calls @ref accxx::get_cuda_error_handler() with a generated @c std::error_code object.
#define ACCXX_CUDA_CALL(...)                                      \
  do                                                              \
  {                                                               \
    auto status = (__VA_ARGS__);                                  \
    if (status != cudaSuccess)                                    \
    {                                                             \
      cudaGetLastError();                                         \
      ::accxx::get_cuda_error_handler()(make_error_code(status)); \
    }                                                             \
  } while (false)

/// @def ACCXX_CUDA_CALL(...)
/// Calls <tt>...</tt> and checks the return value. If the return value does not indicate success,
/// it calls @ref accxx::get_noexcept_cuda_error_handler() with a generated @c std::error_code
/// object.
#define ACCXX_CUDA_CALL_NOEXCEPT(...)                                      \
  do                                                                       \
  {                                                                        \
    auto status = (__VA_ARGS__);                                           \
    if (status != cudaSuccess)                                             \
    {                                                                      \
      cudaGetLastError();                                                  \
      ::accxx::get_noexcept_cuda_error_handler()(make_error_code(status)); \
    }                                                                      \
  } while (false)

#endif
