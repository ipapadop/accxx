# accxx
A C++14 library for simplifying the use of accelerators (CUDA and OpenCL).

## Motivation

This library is a collection of tools for integrating CUDA and OpenCL code as seemlessly as possible with modern C++ codebases.

## Features

- OpenCL error code to string conversion (`accxx/cl/cl_error_code.hpp`)
- Support for transforming CUDA and OpenCL error codes to [`std::error_code`](https://en.cppreference.com/w/cpp/error/error_code) (`accxx/cuda/system_error.hpp`, `accxx/cl/system_error.hpp`)

## Build

accxx has been tested with [CMake](https://cmake.org/) 3.17, but theoretically 3.8 and higher is supported.

If you need to compile with OpenCL, then you will need OpenCL headers and an OpenCL library.

If you need to compile with CUDA, then you will need to have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.

You can build with

```bash
$ git clone https://github.com/ipapadop/accxx
$ export ACCXX_SRC=`pwd`/accxx
$ mkdir accxx-build; cd accxx-build
$ cmake $ACCXX_SRC
$ make -j
```

If CUDA is not found, you can point it to the right CUDA installation with

```bash
$ cmake $ACCXX_SRC -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

If your compiler is too new, the CUDA compiler, `nvcc` may not be able to compile CUDA programs. You can instruct `nvcc` to use your compiler of choice with (e.g., GCC 8 on Ubuntu)

```bash
$ cmake $ACCXX_SRC -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-8
```

## Install

Installation can be accomplished with

```bash
$ make documentation
$ make install
```

## License

```
Copyright (c) 2019-2020 Yiannis Papadopoulos

Distributed under the terms of the MIT License.

(See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
```
