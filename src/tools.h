#ifndef TOOLS_H
#define TOOLS_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sys/time.h>

#include "util.h"

double gettime();

const char* get_error_string(cl_int err);

/** Convert the kernel file into a string */
int convertToString(const char* filename, string& s);

/** Getting platforms and choose an available one */
int getPlatform(cl_platform_id& platform, int id);

/** Query the platform and choose the first device given a device type */
cl_device_id* getCl_device_id(cl_platform_id& platform, char* device_type);

void load(const char* srcdir, smat_t& R, bool ifALS, bool with_weights);

void initial_col(mat_t& X, long k, long n);

#endif
