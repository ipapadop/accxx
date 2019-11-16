/** @file */
/*
 * Copyright (c) 2019 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#include "accxx/cl/system_error.hpp"

namespace {

/// @c std::error_category specialization for CUDA errors.
struct opencl_error_category_t : std::error_category
{
  const char* name() const noexcept override
  {
    return "clError";
  }

  std::string message(int ev) const override
  {
    return accxx::cl_error_code_name(ev);
  }
};

const opencl_error_category_t opencl_error_category;

} // anonymous namespace

namespace accxx {

std::error_code make_error_code(accxx::cl_error_code code)
{
  return {static_cast<int>(code), opencl_error_category};
}

} // namespace accxx
