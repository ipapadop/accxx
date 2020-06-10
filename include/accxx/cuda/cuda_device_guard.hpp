/** @file */
/*
 * Copyright (c) 2019-2020 Yiannis Papadopoulos
 *
 * Distributed under the terms of the MIT License.
 *
 * (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
 */

#ifndef ACCXX_CUDA_CUDA_DEVICE_GUARD_HPP
#define ACCXX_CUDA_CUDA_DEVICE_GUARD_HPP

#include <cuda_runtime_api.h>

#include "accxx/cuda/call.hpp"

namespace accxx {

/**
 * RAII guard that switches to a given device in its constructor, and restores back to the previous
 * device upon destruction.
 */
class cuda_device_guard
{
  int m_saved_device_id{-1};

public:
  /// Switches the current CUDA device to the device with ID @p device_id.
  explicit cuda_device_guard(int device_id)
  {
    int current_device_id{};
    ACCXX_CUDA_CALL(cudaGetDevice(&current_device_id));
    if (current_device_id != device_id)
    {
      m_saved_device_id = current_device_id;
      ACCXX_CUDA_CALL(cudaSetDevice(device_id));
    }
  }

  cuda_device_guard(cuda_device_guard const&) = delete;
  cuda_device_guard& operator=(cuda_device_guard const&) = delete;

  ~cuda_device_guard()
  {
    if (m_saved_device_id != -1)
    {
      ACCXX_CUDA_CALL_NOEXCEPT(cudaSetDevice(m_saved_device_id));
    }
  }
};

} // namespace accxx

#endif
