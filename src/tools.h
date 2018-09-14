#ifndef TOOLS_H
#define TOOLS_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <sys/timeb.h>

#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <string>
#include <fstream>

#include "pmf_util.h"

const char* get_error_string(cl_int err);

/** Convert the kernel file into a string */
int convertToString(const char* filename, std::string& s);

int getPlatform(cl_platform_id& platform, int id);
cl_device_id* getDevice(cl_platform_id& platform, char* device_type);

void print_all_the_info();
void print_all_the_platforms();
int report_device(cl_device_id device_id);

void load(const char* srcdir, smat_t& R, bool ifALS, bool with_weights);

void initial_col(mat_t& X, long k, long n);

#endif
