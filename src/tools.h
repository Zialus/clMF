#ifndef TOOLS_H
#define TOOLS_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_SILENCE_DEPRECATION
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <string>
#include <fstream>

#include "pmf_util.h"
#include "pmf.h"

#define CL_CHECK(res) \
    {if (res != CL_SUCCESS) {fprintf(stderr,"Error \"%s\" (%d) in file %s on line %d\n", \
        get_error_string(res), res, __FILE__,__LINE__); abort();}}

const char* get_error_string(cl_int err);

void convertToString(const char* filename, std::string& s);

cl_platform_id getPlatform(int id);

cl_device_id* getDevices(cl_platform_id& platform, char* device_type);

void print_all_the_info();

void print_platform_info(cl_platform_id* platforms, unsigned int id);

void print_device_info(cl_device_id* devices, unsigned int j);

int report_device(cl_device_id device_id);

void load(const char* srcdir, smat_t& R, bool ifALS, bool with_weights);

void initial_col(mat_t& X, long k, long n);

void exit_with_help();

parameter parse_command_line(int argc, char** argv);

#endif //TOOLS_H
