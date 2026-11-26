# neongammon — top-level CMakeLists.txt for the pip (scikit-build-core) build.
#
# This file delegates the compiled-extension build to cpp/CMakeLists.txt (the
# engine build) and then adds the install steps that place the pybind11
# module, model weights, and bearoff database into the wheel.

cmake_minimum_required(VERSION 3.18)
project(neongammon LANGUAGES CXX)

# Don't build the benchmark / profiler executables in the wheel — just the
# pybind11 module. Dev builds leave this ON (the default in cpp/CMakeLists.txt).
set(BGBOT_BUILD_APPS OFF CACHE BOOL "Build benchmark/profiler executables")

include("${CMAKE_CURRENT_SOURCE_DIR}/cpp/CMakeLists.txt")

# Install the Python extension at the wheel root so `import bgbot_cpp` works.
# scikit-build-core's default install prefix is the site-packages root.
if(TARGET bgbot_cpp)
    install(TARGETS bgbot_cpp
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .)
endif()

# Read the same explicit paths scikit-build-core uses for its sdist.
find_package(Python 3.10 COMPONENTS Interpreter REQUIRED)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    pyproject.toml cmake/model_weights.py)
execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/model_weights.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE weights_status
    OUTPUT_VARIABLE model_weights
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT weights_status EQUAL 0)
    message(FATAL_ERROR "Cannot read the model weight manifest in pyproject.toml")
endif()
install(FILES ${model_weights} DESTINATION neongammon/_assets/models)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/bearoff_1sided.db"
    DESTINATION neongammon/_assets/data)
