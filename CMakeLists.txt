# Minimal version of CMake
cmake_minimum_required(VERSION 3.9)

# Define project name
project(clMF)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/exec/")

#set(CMAKE_VERBOSE_MAKEFILE ON)

# Source code of application
file(GLOB source_src "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB source_hdr "${PROJECT_SOURCE_DIR}/src/*.h")


# Compiler flags to enable all warnings
#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
#elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
#endif()

# Silence warning C4996 on Microsoft Compilers
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif ()

# Set up executable
add_executable(clMF ${source_src} ${source_hdr})
target_link_libraries(clMF OpenMP::OpenMP_CXX OpenCL::OpenCL)


enable_testing()
add_test(NAME simple COMMAND $<TARGET_FILE:clMF> -k 10 -t 10 -l 0.05 ../data/simple)
add_test(NAME toy-example COMMAND $<TARGET_FILE:clMF> -k 10 -t 10 -l 0.05 ../data/toy-example)
