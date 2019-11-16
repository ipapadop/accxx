if (OpenCL_FOUND)
  add_subdirectory(cl)
endif()

if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
  set(CUDA_SOURCE_FILES ${CUDA_SOURCE_FILES} PARENT_SCOPE)
endif()

set(SOURCE_FILES ${SOURCE_FILES} PARENT_SCOPE)
