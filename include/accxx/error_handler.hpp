/** @file */
/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#ifndef ACCXX_ERROR_HANDLER_HPP
#define ACCXX_ERROR_HANDLER_HPP

#include <functional>
#include <system_error>

namespace accxx {

/// Error handling function type
using error_handler = std::function<void(std::error_code)>;

} // namespace accxx

#endif
