# accxx
A C++14 library for simplifying the use of accelerators (CUDA and OpenCL).

## Motivation

This library is a collection of tools for integrating CUDA and OpenCL code as seemlessly as possible with modern C++ codebases.

## Features

- OpenCL error code to string conversion (`accxx/cl/cl_error_code.hpp`)
- Support for transforming CUDA and OpenCL error codes to [`std::error_code`](https://en.cppreference.com/w/cpp/error/error_code) (`accxx/cuda/system_error.hpp`, `accxx/cl/system_error.hpp`)

## License

```
Copyright (c) 2019 Yiannis Papadopoulos

Distributed under the terms of the MIT License.

(See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
```
