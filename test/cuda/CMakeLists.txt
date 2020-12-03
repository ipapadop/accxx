set(SOURCES
  call.cu
  cuda_device_guard.cpp
  error_handler.cpp
  main.cpp
  system_error.cpp)

set_source_files_properties(${SOURCES}
    PROPERTIES
      LANGUAGE CUDA)

add_executable(cuda_tests ${SOURCES})

target_link_libraries(cuda_tests
  PRIVATE
    accxx
    Catch2::Catch2)

add_test(
  NAME
    UnitTests
  COMMAND
    cuda_tests)
