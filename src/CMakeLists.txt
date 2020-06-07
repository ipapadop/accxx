if(ACCXX_WITH_CUDA)
  add_subdirectory(cuda)
  set(CUDA_SOURCE_FILES ${CUDA_SOURCE_FILES} PARENT_SCOPE)
endif()

if (ACCXX_WITH_OPENCL)
  add_subdirectory(cl)
endif()

set(SOURCE_FILES ${SOURCE_FILES} PARENT_SCOPE)
