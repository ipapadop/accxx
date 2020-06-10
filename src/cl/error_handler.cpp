/** @file */
/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#include <cstdlib>
#include <iostream>
#include <utility>

#include "accxx/cuda/error_handler.hpp"

namespace accxx {

namespace {

/**
 * @brief Prints the error and exits.
 */
void default_handler(std::error_code error_code)
{
  std::cerr << error_code.category().name() << ": " << error_code.message() << " ("
            << error_code.value() << ")\n";
  std::exit(error_code.value());
}

// handler for all errors
error_handler handler{default_handler};

// handler for errors that execute within under noexcept
error_handler noexcept_handler{default_handler};

} // namespace

error_handler set_cl_error_handler(error_handler new_handler) noexcept
{
  return std::exchange(handler, new_handler);
}

error_handler get_cl_error_handler() noexcept
{
  return handler;
}

error_handler set_noexcept_cl_error_handler(error_handler new_handler) noexcept
{
  return std::exchange(noexcept_handler, new_handler);
}

error_handler get_noexcept_cl_error_handler() noexcept
{
  return noexcept_handler;
}

} // namespace accxx
