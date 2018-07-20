#include "cj.h"

using namespace std;

int convertToString(const char* filename, string& s) {
    size_t size;
    char* str;
    fstream f(filename, (fstream::in | fstream::binary));

    if (f.is_open()) {
        size_t fileSize;
        f.seekg(0, fstream::end);
        size = fileSize = (size_t) f.tellg();
        f.seekg(0, fstream::beg);
        str = new char[size + 1];
        if (!str) {
            f.close();
            return 0;
        }
        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    cout << "Error:failed to open file:" << filename << "\n";
    return -1;
}

int getPlatform(cl_platform_id& platform, int id) {
    platform = NULL;
    cl_uint numPlatforms;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS) {
        cout << "ERROR:Getting platforms!\n";
        return -1;
    }
    if (numPlatforms > 0) {
        cl_platform_id* platforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (status != CL_SUCCESS) {
            cout << "ERROR:Getting platform IDs!\n";
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
    cl_device_id* devices = NULL;

    if ((device_type[0] == 'm') && (device_type[1] == 'i') && (device_type[2] == 'c')) {
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &numDevices);
    } else if ((device_type[0] == 'c') && (device_type[1] == 'p') && (device_type[2] == 'u')) {
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
    } else if ((device_type[0] == 'g') && (device_type[1] == 'p') && (device_type[2] == 'u')) {
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    }

    if (status != CL_SUCCESS) {
        cout << "ERROR:Getting numDevices from clGetDeviceIDs!\n";
        exit(1);
    }

    if (numDevices > 0) {
        devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
        if ((device_type[0] == 'c') && (device_type[1] == 'p') && (device_type[2] == 'u')) {
            status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
        } else if ((device_type[0] == 'm') && (device_type[1] == 'i') && (device_type[2] == 'c')) {
            status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, numDevices, devices, NULL);
        } else if ((device_type[0] == 'g') && (device_type[1] == 'p') && (device_type[2] == 'u')) {
            status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
        }
    }

    if (status != CL_SUCCESS) {
        cout << "ERROR:Getting devices from clGetDeviceIDs!\n";
        exit(1);
    }
    return devices;
}

void load(const char* srcdir, smat_t& R, bool ifALS, bool with_weights) {
    char filename[1024], buf[1024];
    sprintf(filename, "%s/meta", srcdir);
    FILE* fp = fopen(filename, "r");
    long m, n, nnz;

    fscanf(fp, "%ld %ld", &m, &n);
    fscanf(fp, "%ld %s", &nnz, buf);
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
