cmake_minimum_required(VERSION 3.14)

find_package(deal.II 9.7.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()

if(NOT ((DEAL_II_WITH_PETSC AND NOT DEAL_II_PETSC_WITH_COMPLEX) OR DEAL_II_WITH_TRILINOS) OR NOT DEAL_II_WITH_P4EST)
  message(FATAL_ERROR "
Error! This project requires a deal.II library configured with either:
    DEAL_II_WITH_PETSC = ON
    DEAL_II_PETSC_WITH_COMPLEX = OFF
    DEAL_II_WITH_P4EST = ON
or
    DEAL_II_WITH_TRILINOS = ON
    DEAL_II_WITH_P4EST = ON
The library at ${DEAL_II_PATH} was configured with:
    DEAL_II_WITH_PETSC = ${DEAL_II_WITH_PETSC}
    DEAL_II_PETSC_WITH_COMPLEX = ${DEAL_II_PETSC_WITH_COMPLEX}
    DEAL_II_WITH_P4EST = ${DEAL_II_WITH_P4EST}
    DEAL_II_WITH_TRILINOS = ${DEAL_II_WITH_TRILINOS}
At least one of the required combinations must be available.")
endif()

deal_ii_initialize_cached_variables()
project(pmf-form)


include(FetchContent)
include(CTest)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.11.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(elasticity elasticity/elasticity.cc)
target_include_directories(elasticity PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_executable(laplace laplace/laplace.cc)
target_include_directories(laplace PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_executable(stokes stokes/stokes.cc)

deal_ii_setup_target(elasticity)
deal_ii_setup_target(laplace)
deal_ii_setup_target(stokes)

file(GLOB_RECURSE PROJECT_FORMAT_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)
list(FILTER PROJECT_FORMAT_FILES EXCLUDE REGEX "/build/")
add_custom_target(indent
  COMMAND clang-format -i ${PROJECT_FORMAT_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Formatting project C++ files with clang-format"
  VERBATIM
)

if(BUILD_TESTING)
  add_executable(pmf_form_tests tests/test_main.cc)
  deal_ii_setup_target(pmf_form_tests)
  target_link_libraries(pmf_form_tests Catch2::Catch2)
  target_include_directories(pmf_form_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(
    NAME pmf_form_tests
    COMMAND mpiexec -n 1 $<TARGET_FILE:pmf_form_tests>
  )
endif()
