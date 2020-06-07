if(ACCXX_WITH_CUDA)
  add_subdirectory(cuda)
endif()

if(ACCXX_WITH_OPENCL)
  add_subdirectory(cl)
endif()
