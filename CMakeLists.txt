cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(accxx
  VERSION 1.0.0
  LANGUAGES CXX)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

# options

option(ACCXX_WITH_CUDA "Compile with CUDA supprot" ON)
option(ACCXX_WITH_OPENCL "Compile with OpenCL supprot" ON)

# dependencies

add_subdirectory(external)

if(ACCXX_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_HOST_COMPILER)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 14)
      set(CMAKE_CUDA_STANDARD_REQUIRED True)
    endif()
  else()
    message(STATUS "No CUDA support")
    unset(ACCXX_WITH_CUDA CACHE)
  endif()
endif()

if(ACCXX_WITH_OPENCL)
  find_package(OpenCL)
  if (NOT OpenCL_FOUND)
    message(STATUS "No OpenCL support")
    unset(ACCXX_WITH_OPENCL CACHE)
  endif()
endif()

# targets and properties

add_subdirectory(src)

if(ACCXX_WITH_CUDA)
  set_source_files_properties(${CUDA_SOURCE_FILES}
    PROPERTIES
      LANGUAGE CUDA)
  set(SOURCE_FILES ${SOURCE_FILES} ${CUDA_SOURCE_FILES})
endif()

add_library(accxx SHARED ${SOURCE_FILES})
add_library(accxx::accxx ALIAS accxx)
target_include_directories(accxx
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(accxx PRIVATE cxx_std_14)
#target_compile_options(accxx PRIVATE
#  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
#    -Wall -Wextra -Wpedantic>)
if(ACCXX_WITH_OPENCL)
  target_compile_definitions(accxx PUBLIC CL_TARGET_OPENCL_VERSION=120)
  target_link_libraries(accxx PUBLIC OpenCL::OpenCL)
endif()

# installation

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/accxx)

install(TARGETS accxx
  EXPORT accxx-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT accxx-targets
  FILE
    accxx.cmake
  NAMESPACE
    accxx::
  DESTINATION
    ${INSTALL_CONFIGDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/accxxConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/accxxConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/accxxConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/accxxConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/accxxConfigVersion.cmake
  DESTINATION ${INSTALL_CONFIGDIR})

# exporting

export(EXPORT accxx-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/accxxTargets.cmake
    NAMESPACE accxx::)

export(PACKAGE accxx)

# documentation

find_package(Doxygen)
option(ACCXX_BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ${DOXYGEN_FOUND})

if(ACCXX_BUILD_DOCUMENTATION)
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

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(test)
endif()
