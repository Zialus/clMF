# Minimal version of CMake
cmake_minimum_required (VERSION 3.9)

# Define project name
project(clMF)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/exec/")

#set(CMAKE_VERBOSE_MAKEFILE ON)

#set(WARNINGS "-Wno-deprecated-declarations")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNINGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNINGS}")


# Source code of application
file(GLOB source_src "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB source_hdr "${PROJECT_SOURCE_DIR}/src/*.h")


# Compiler flags
#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
#elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
#endif()


# Set up executable
add_executable(clMF ${source_src} ${source_hdr})
target_link_libraries(clMF OpenMP::OpenMP_CXX OpenCL::OpenCL)


enable_testing()
add_test(simple ../exec/clMF -k 10 -t 5 -T 1 -l 0.05 -n 10 -nBlocks 32 -nThreadsPerBlock 256 ../data/simple)
add_test(toy-example ../exec/clMF -k 10 -t 5 -T 1 -l 0.05 -n 10 -nBlocks 32 -nThreadsPerBlock 256 ../data/toy-example)
