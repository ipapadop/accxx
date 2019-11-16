cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(accxx
  VERSION 1.0.0
  LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
find_package(OpenCL)

add_subdirectory(external)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_CUDA_COMPILER)
  project(accxx LANGUAGES CUDA)
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
  endif()
else()
  message(STATUS "No CUDA support")
endif()

if (NOT OpenCL_FOUND)
  message(STATUS "No OpenCL support")
endif()

add_subdirectory(src)

if(CMAKE_CUDA_COMPILER)
  set_source_files_properties(${CUDA_SOURCE_FILES}
    PROPERTIES
      LANGUAGE CUDA)
  set(SOURCE_FILES ${SOURCE_FILES} ${CUDA_SOURCE_FILES})
endif()

add_library(accxx ${SOURCE_FILES})
target_include_directories(accxx
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(accxx PUBLIC cxx_std_14)
if(OpenCL_FOUND)
  target_link_libraries(accxx PUBLIC OpenCL::OpenCL)
endif()
add_library(accxx::accxx ALIAS accxx)

# documentation
find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ${DOXYGEN_FOUND})

if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(documentation
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
endif()

# testing
include(CTest)
enable_testing()
add_subdirectory(test)
