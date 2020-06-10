/** @file */
/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#ifndef ACCXX_CL_CALL_HPP
#define ACCXX_CL_CALL_HPP

#include <CL/cl.h>

#include "accxx/cl/error_handler.hpp"
#include "accxx/cl/system_error.hpp"

/// @def ACCXX_CL_CALL(...)
/// Calls <tt>...</tt> and checks the return value. If the return value does not indicate success,
/// it calls @ref accxx::get_cl_error_handler() with a generated @c std::error_code object.
#define ACCXX_CL_CALL(...)                                                              \
  do                                                                                    \
  {                                                                                     \
    auto status = (__VA_ARGS__);                                                        \
    if (status != CL_SUCCESS)                                                           \
    {                                                                                   \
      ::accxx::get_cl_error_handler()(make_error_code(::accxx::cl_error_code(status))); \
    }                                                                                   \
  } while (false)

/// @def ACCXX_CL_CALL(...)
/// Calls <tt>...</tt> and checks the return value. If the return value does not indicate success,
/// it calls @ref accxx::get_noexcept_cl_error_handler() with a generated @c std::error_code
/// object.
#define ACCXX_CL_CALL_NOEXCEPT(...)                                                              \
  do                                                                                             \
  {                                                                                              \
    auto status = (__VA_ARGS__);                                                                 \
    if (status != cudaSuccess)                                                                   \
    {                                                                                            \
      ::accxx::get_noexcept_cl_error_handler()(make_error_code(::accxx::cl_error_code(status))); \
    }                                                                                            \
  } while (false)

#endif
