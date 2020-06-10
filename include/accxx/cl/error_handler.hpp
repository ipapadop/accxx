/** @file */
/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#ifndef ACCXX_CL_ERROR_HANDLER_HPP
#define ACCXX_CL_ERROR_HANDLER_HPP

#include "accxx/error_handler.hpp"

namespace accxx {

/// Makes @p new_handler the global CUDA error handler function and returns the previously installed
/// CUDA error handler.
error_handler set_cl_error_handler(error_handler new_handler) noexcept;

/// Returns the installed global CUDA error handler.
error_handler get_cl_error_handler() noexcept;

/// Makes @p new_handler the global CUDA error handler function for @c noexcept functions and
/// returns the previously installed CUDA error handler.
error_handler set_noexcept_cl_error_handler(error_handler new_handler) noexcept;

/// Returns the installed global CUDA error handler for @c noexcept functions.
error_handler get_noexcept_cl_error_handler() noexcept;

} // namespace accxx

#endif
