if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
endif()

if(OpenCL_FOUND)
  add_subdirectory(cl)
endif()
