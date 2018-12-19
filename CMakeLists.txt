# Minimal version of CMake
cmake_minimum_required(VERSION 3.9)

# Define project name
project(clMF)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_VERBOSE_MAKEFILE ON)

# Compiler flags to enable more warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif ()

# Compiler flags to enable all warnings
#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
#elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
#elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
#endif ()

# Compiler flags to silence some of the warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif ()

# Silence warning C4996 on Microsoft Compilers
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif ()

# Source code
file(GLOB source_src "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB source_hdr "${PROJECT_SOURCE_DIR}/src/*.h")

# Executable and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/exec/")
add_executable(clMF ${source_src} ${source_hdr})
target_link_libraries(clMF OpenCL::OpenCL OpenMP::OpenMP_CXX)

# Tests
enable_testing()
add_test(NAME simple COMMAND $<TARGET_FILE:clMF> -k 10 -t 10 -l 0.05 -P 0 -d 1 -q 1 -r 1 -p 1 -nBlocks $ENV{BLOCKS_FLAG} -nThreadsPerBlock $ENV{THREADS_FLAG} ${CMAKE_SOURCE_DIR}/data/simple)
add_test(NAME simple_rolled COMMAND $<TARGET_FILE:clMF> -k 10 -t 10 -l 0.05 -P 0 -d 1 -q 1 -r 1 -p 1 -nBlocks $ENV{BLOCKS_FLAG} -nThreadsPerBlock $ENV{THREADS_FLAG} -c ${CMAKE_SOURCE_DIR}/kcode/ALS_rolled.cl ${CMAKE_SOURCE_DIR}/data/simple)
#add_test(NAME toy-example COMMAND $<TARGET_FILE:clMF> -k 10 -t 10 -l 0.05 -P 0 -d 1 -q 1 -r 1 -p 1 -nBlocks $ENV{BLOCKS_FLAG} -nThreadsPerBlock $ENV{THREADS_FLAG} ${CMAKE_SOURCE_DIR}/data/toy-example)
