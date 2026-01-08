cmake_minimum_required(VERSION 3.15...3.27)

# scikit-build-core defines these. keep a fallback for non-skbuild usage.
if(DEFINED SKBUILD_PROJECT_NAME)
  project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)
else()
  project(commander4 VERSION 0.1.0 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(DEFINED ENV{CMDR4_OPTIMIZATION})
  set(OPTFLAGS "$ENV{CMDR4_OPTIMIZATION}")
else()
  set(OPTFLAGS "native-strip")
endif()

# Match the macros used by src/lib_cpp/cmdr4_support.cc
# - PKGNAME must expand to an identifier for PYBIND11_MODULE(PKGNAME, m)
# - PKGVERSION is stringified inside the C++ code.
set(PKGNAME _cmdr4_backend)
set(PKGVERSION ${PROJECT_VERSION})

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# --- Sanity checks (match previous Meson behavior) ---
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/ducc0/src/ducc0/infra/mav.cc")
  message(FATAL_ERROR "ducc0 submodule missing! Run: git submodule update --init --recursive")
endif()

set(CXX_SOURCES
  src/lib_cpp/cmdr4_support.cc
  external/ducc0/src/ducc0/infra/string_utils.cc
  external/ducc0/src/ducc0/infra/threading.cc
  external/ducc0/src/ducc0/infra/mav.cc
)

### ctypes libraries ###
# Native C ABI helpers loaded via ctypes.
# Keep these in one shared library
# To add new ctypes helpers: drop a new .cpp into src/lib_cpp/ctypes/.
file(GLOB CMDR4_CTYPES_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/lib_cpp/ctypes/*.cpp"
)
list(SORT CMDR4_CTYPES_SOURCES)

add_library(cmdr4_ctypes SHARED ${CMDR4_CTYPES_SOURCES})
set_target_properties(cmdr4_ctypes PROPERTIES
  OUTPUT_NAME "cmdr4_ctypes"
  PREFIX ""
)

if(DEFINED ENV{CMDR4_USE_NANOBIND})
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR
  )
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(${PKGNAME} NB_SUPPRESS_WARNINGS NOMINSIZE ${CXX_SOURCES})
  target_compile_definitions(${PKGNAME} PRIVATE CMDR4_USE_NANOBIND)
else()
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(${PKGNAME} MODULE ${CXX_SOURCES})
endif()

target_include_directories(${PKGNAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/external/ducc0/src
)

target_include_directories(cmdr4_ctypes PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lib_cpp
)

target_compile_definitions(${PKGNAME} PRIVATE PKGNAME=${PKGNAME} PKGVERSION=${PKGVERSION})

### Architecture flags ###
if(NOT MSVC)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc|ppc64|powerpc|powerpc64")
    set(CMDR4_ARCH_FLAGS "-mtune=native" CACHE STRING "Compiler flags for specifying target architecture.")
  elseif(CMAKE_SYSTEM_NAME MATCHES "Emscripten")
    set(CMDR4_ARCH_FLAGS "" CACHE STRING "Compiler flags for specifying target architecture.")
  else()
    set(CMDR4_ARCH_FLAGS "-march=native" CACHE STRING "Compiler flags for specifying target architecture.")
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMDR4_WARN_FLAGS
    -Wfatal-errors -Wfloat-conversion -W -Wall -Wstrict-aliasing
    -Wwrite-strings -Wredundant-decls -Woverloaded-virtual -Wcast-qual
    -Wcast-align -Wpointer-arith -Wnon-virtual-dtor
    -Wzero-as-null-pointer-constant
  )
  set(CMDR4_OPT_FLAGS -O3 -ffast-math)
  set(CMDR4_NOOPT_FLAGS -O0)
  set(CMDR4_DEBUG_FLAGS -g)
  set(CMDR4_STRIP_FLAGS -s)
endif()

target_compile_options(${PKGNAME} PRIVATE ${CMDR4_WARN_FLAGS})
target_compile_options(cmdr4_ctypes PRIVATE ${CMDR4_WARN_FLAGS})

if(OPTFLAGS MATCHES "^native")
  target_compile_options(${PKGNAME} PRIVATE ${CMDR4_ARCH_FLAGS})
  target_compile_options(cmdr4_ctypes PRIVATE ${CMDR4_ARCH_FLAGS})
endif()
if(OPTFLAGS MATCHES "-debug$")
  target_compile_options(${PKGNAME} PRIVATE ${CMDR4_DEBUG_FLAGS})
  target_compile_options(cmdr4_ctypes PRIVATE ${CMDR4_DEBUG_FLAGS})
elseif(OPTFLAGS MATCHES "-strip$")
  target_link_options(${PKGNAME} PRIVATE ${CMDR4_STRIP_FLAGS})
  target_link_options(cmdr4_ctypes PRIVATE ${CMDR4_STRIP_FLAGS})
endif()
if(OPTFLAGS MATCHES "^none")
  target_compile_options(${PKGNAME} PRIVATE ${CMDR4_NOOPT_FLAGS})
  target_compile_options(cmdr4_ctypes PRIVATE ${CMDR4_NOOPT_FLAGS})
else()
  target_compile_options(${PKGNAME} PRIVATE ${CMDR4_OPT_FLAGS})
  target_compile_options(cmdr4_ctypes PRIVATE ${CMDR4_OPT_FLAGS})
endif()

find_package(Threads REQUIRED)
target_link_libraries(${PKGNAME} PRIVATE Threads::Threads)
target_link_libraries(cmdr4_ctypes PRIVATE Threads::Threads)

# ctypes sources may include <omp.h>. Link OpenMP if available.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(cmdr4_ctypes PRIVATE OpenMP::OpenMP_CXX)
endif()

if(UNIX AND NOT APPLE)
  target_link_libraries(cmdr4_ctypes PRIVATE m)
endif()

if(DEFINED ENV{CMDR4_CFLAGS})
  separate_arguments(CFLAGS_TMP NATIVE_COMMAND "$ENV{CMDR4_CFLAGS}")
  target_compile_options(${PKGNAME} PRIVATE ${CFLAGS_TMP})
  target_compile_options(cmdr4_ctypes PRIVATE ${CFLAGS_TMP})
endif()
if(DEFINED ENV{CMDR4_LFLAGS})
  separate_arguments(LFLAGS_TMP NATIVE_COMMAND "$ENV{CMDR4_LFLAGS}")
  target_link_options(${PKGNAME} PRIVATE ${LFLAGS_TMP})
  target_link_options(cmdr4_ctypes PRIVATE ${LFLAGS_TMP})
endif()
if(DEFINED ENV{CMDR4_FLAGS})
  separate_arguments(FLAGS_TMP NATIVE_COMMAND "$ENV{CMDR4_FLAGS}")
  target_compile_options(${PKGNAME} PRIVATE ${FLAGS_TMP})
  target_link_options(${PKGNAME} PRIVATE ${FLAGS_TMP})
  target_compile_options(cmdr4_ctypes PRIVATE ${FLAGS_TMP})
  target_link_options(cmdr4_ctypes PRIVATE ${FLAGS_TMP})
endif()

target_compile_features(${PKGNAME} PRIVATE cxx_std_17)
target_compile_features(cmdr4_ctypes PRIVATE cxx_std_17)

set_property(TARGET ${PKGNAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
set_property(TARGET cmdr4_ctypes PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

install(TARGETS ${PKGNAME} LIBRARY DESTINATION commander4)
install(TARGETS cmdr4_ctypes LIBRARY DESTINATION commander4/_libs)

### Stub generation ###
# Runs stubgen during the CMake install step (which scikit-build-core runs when assembling
# wheels/editable installs).
# Output is written into the source tree (src-layout) to keep IDE support for editable installs.
option(CMDR4_GENERATE_STUBS "Generate .pyi stubs during install" ON)

if(CMDR4_GENERATE_STUBS)
  if(DEFINED ENV{CMDR4_USE_NANOBIND})
    message(STATUS "CMDR4_USE_NANOBIND set; skipping pybind11-stubgen")
  else()
    install(CODE "
      message(STATUS \"Generating stubs for commander4._cmdr4_backend...\")
      execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E env
                \"PYTHONPATH=${CMAKE_INSTALL_PREFIX}\"
                \"${Python_EXECUTABLE}\" -m pybind11_stubgen
                commander4._cmdr4_backend
                --output-dir \"${CMAKE_CURRENT_SOURCE_DIR}/src\"
                --root-suffix \"\"
        RESULT_VARIABLE _cmdr4_stubgen_res
      )
      if(NOT _cmdr4_stubgen_res EQUAL 0)
        message(WARNING \"pybind11_stubgen failed with exit code: ${_cmdr4_stubgen_res} (continuing without regenerated stubs)\")
      endif()
    ")
  endif()
endif()
