#include "tools.h"

const char* get_error_string(cl_int err){
    switch(err){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";

        default: return "Unknown OpenCL error";
    }
}

int convertToString(const char* filename, std::string& s) {
    char* str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if (f.is_open()) {
        size_t fileSize;
        size_t size;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t) f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size + 1];
        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    std::cout << "Error:failed to open file:" << filename << "\n";
    return -1;
}

int getPlatform(cl_platform_id& platform, int id) {
    platform = nullptr;
    cl_uint numPlatforms;
    cl_int status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (status != CL_SUCCESS) {
        std::cout << "ERROR:Getting platforms!\n";
        return -1;
    }
    if (numPlatforms > 0) {
        cl_platform_id* platforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
        if (status != CL_SUCCESS) {
            std::cout << "ERROR:Getting platform IDs!\n";
            free(platforms);
            return -1;
        }
        platform = platforms[id];
        free(platforms);
        return 0;
    } else {
        return -1;
    }
}

cl_device_id* getCl_device_id(cl_platform_id& platform, char* device_type) {
    cl_uint numDevices = 0;
    cl_int status = 0;
    cl_device_id* devices = nullptr;

    if ((device_type[0] == 'm') && (device_type[1] == 'i') && (device_type[2] == 'c')) {
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 0, nullptr, &numDevices);
    } else if ((device_type[0] == 'c') && (device_type[1] == 'p') && (device_type[2] == 'u')) {
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &numDevices);
    } else if ((device_type[0] == 'g') && (device_type[1] == 'p') && (device_type[2] == 'u')) {
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    }

    if (status != CL_SUCCESS) {
        std::cout << "ERROR:Getting numDevices from clGetDeviceIDs!\n";
        exit(1);
    }

    if (numDevices > 0) {
        devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
        if ((device_type[0] == 'c') && (device_type[1] == 'p') && (device_type[2] == 'u')) {
            status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, nullptr);
        } else if ((device_type[0] == 'm') && (device_type[1] == 'i') && (device_type[2] == 'c')) {
            status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, numDevices, devices, nullptr);
        } else if ((device_type[0] == 'g') && (device_type[1] == 'p') && (device_type[2] == 'u')) {
            status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr);
        }
    }

    if (status != CL_SUCCESS) {
        std::cout << "ERROR:Getting devices from clGetDeviceIDs!\n";
        exit(1);
    }
    return devices;
}

void load(const char* srcdir, smat_t& R, bool ifALS, bool with_weights) {
    char filename[1024], buf[1024];
    sprintf(filename, "%s/meta", srcdir);
    FILE* fp = fopen(filename, "r");
    if (fp == nullptr) {
        printf("Can't open input file.\n");
        exit(1);
    }
    unsigned m, n, nnz;
    fscanf(fp, "%u %u", &m, &n);
    fscanf(fp, "%u %s", &nnz, buf);
    sprintf(filename, "%s/%s", srcdir, buf);
    R.load(m, n, nnz, filename, ifALS, with_weights);
    fclose(fp);
}

void initial_col(mat_t& X, long k, long n) {
    X = mat_t(k, vec_t(n));
    srand(0L);
    //srand48(0L);
    long i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            X[j][i] = 0.1 * (float(rand()) / RAND_MAX) + 0.001;
            //X[j][i] = 0.1*drand48();
        }
    }
}

double gettime() {
    struct timeval t;
    gettimeofday(&t, nullptr);
    return t.tv_sec + t.tv_usec * 1e-6;
}
