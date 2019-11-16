/** @file */
/*
 * Copyright (c) 2019 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#include "accxx/cuda/system_error.hpp"

namespace {

/// @c std::error_category specialization for CUDA errors.
struct cuda_error_category_t : std::error_category
{
  const char* name() const noexcept override
  {
    return "cudaError";
  }

  std::string message(int ev) const override
  {
    return cudaGetErrorString(static_cast<cudaError_t>(ev));
  }
};

const cuda_error_category_t cuda_error_category;

} // anonymous namespace

std::error_code make_error_code(cudaError_t code)
{
  return {static_cast<int>(code), cuda_error_category};
}
