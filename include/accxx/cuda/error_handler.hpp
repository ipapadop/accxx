/** @file */
/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#ifndef ACCXX_CUDA_ERROR_HANDLER_HPP
#define ACCXX_CUDA_ERROR_HANDLER_HPP

#include <functional>
#include <system_error>

namespace accxx {


using error_handler = std::function<void(std::error_code)>;

/// Makes @p new_handler the global CUDA error handler function and returns the previously installed
/// CUDA error handler.
error_handler set_cuda_error_handler(error_handler new_handler) noexcept;

/// Returns the installed global CUDA error handler.
error_handler get_cuda_error_handler() noexcept;

/// Makes @p new_handler the global CUDA error handler function for @c noexcept functions and
/// returns the previously installed CUDA error handler.
error_handler set_noexcept_cuda_error_handler(error_handler new_handler) noexcept;

/// Returns the installed global CUDA error handler for @c noexcept functions.
error_handler get_noexcept_cuda_error_handler() noexcept;

} // namespace accxx

#endif
