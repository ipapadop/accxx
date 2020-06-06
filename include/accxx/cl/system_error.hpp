/** @file */
/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#ifndef ACCXX_CL_SYSTEM_ERROR_HPP
#define ACCXX_CL_SYSTEM_ERROR_HPP

#include <system_error>

#include "accxx/cl/cl_error_code.hpp"

namespace std {

template<>
struct is_error_code_enum<accxx::cl_error_code> : true_type
{};

} // namespace std

namespace accxx {

/// Creates a @c std::error_code from @p code.
std::error_code make_error_code(accxx::cl_error_code code);

} // namespace accxx

#endif
