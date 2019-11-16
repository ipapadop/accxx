set(SOURCES
  main.cpp
  system_error.cpp)

add_executable(cl_tests ${SOURCES})

target_link_libraries(cl_tests
  PRIVATE
    accxx
    Catch2::Catch2)

add_test(
  NAME
    UnitTests
  COMMAND
    cl_tests)

