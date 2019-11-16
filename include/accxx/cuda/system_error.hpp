/** @file */
/*
 * Copyright (c) 2019 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#ifndef ACCXX_CUDA_SYSTEM_ERROR_HPP
#define ACCXX_CUDA_SYSTEM_ERROR_HPP

#include <system_error>

#include <cuda_runtime_api.h>

namespace std {

template<>
struct is_error_code_enum<cudaError_t> : true_type
{};

} // namespace std

/// Creates a @c std::error_code from @p code.
std::error_code make_error_code(cudaError_t code);

#endif
