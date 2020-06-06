/** @file */
/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#ifndef ACCXX_CL_ERROR_CODE_HPP
#define ACCXX_CL_ERROR_CODE_HPP

#include <CL/opencl.h>

namespace accxx {

/// Wrapper for OpenCL error codes.
class cl_error_code
{
  cl_int m_code{CL_SUCCESS};

public:
  cl_error_code() = default;

  /// Constructs the wrapper with error code @p code.
  constexpr cl_error_code(cl_int code) noexcept : m_code{code}
  {}

  constexpr cl_int* operator&() noexcept
  {
    return &m_code;
  }

  constexpr operator cl_int() const noexcept
  {
    return m_code;
  }

  friend constexpr bool
  operator==(cl_error_code const& x, cl_error_code const& y) noexcept
  {
    return x.m_code == y.m_code;
  }

  friend constexpr bool
  operator!=(cl_error_code const& x, cl_error_code const& y) noexcept
  {
    return !(x == y);
  }

  friend constexpr bool
  operator<(cl_error_code const& x, cl_error_code const& y) noexcept
  {
    return x.m_code < y.m_code;
  }

  friend constexpr bool operator==(int x, cl_error_code const& y) noexcept
  {
    return x == y.m_code;
  }

  friend constexpr bool operator!=(int x, cl_error_code const& y) noexcept
  {
    return !(x == y);
  }

  friend constexpr bool
  operator<(int x, cl_error_code const& y) noexcept
  {
    return x < y.m_code;
  }

  friend constexpr bool operator==(cl_error_code const& x, int y) noexcept
  {
    return x.m_code == y;
  }

  friend constexpr bool operator!=(cl_error_code const& x, int y) noexcept
  {
    return !(x == y);
  }

  friend constexpr bool
  operator<(cl_error_code const& x, int y) noexcept
  {
    return x.m_code < y;
  }
};

/// Returns the string representation of @p code.
const char* cl_error_code_name(int code) noexcept;

} // namespace accxx

#endif
