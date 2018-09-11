#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "pmf.h"
#include "pmf_util.h"
#include "tools.h"

void choldc1(int n, float** a, float* p) {
    unsigned i, j;
    int k;
    float sum;
    for (i = 0; i < n; ++i) {
        for (j = i; j < n; ++j) {
            sum = a[i][j];
            for (k = i - 1; k >= 0; --k) {
                sum -= a[i][k] * a[j][k];
            }
            if (i == j) {
                if (sum <= 0) {
                    printf(" a is not positive definite!\n");
                }
                p[i] = sqrtf(sum);//float square root
            } else {
                a[j][i] = sum / p[i];
            }
        }
    }
}

void choldcsl(int n, float** A) {
    unsigned i, j, k;
    double sum;
    float* p;
    p = (float*) malloc(n * sizeof(float));
    choldc1(n, A, p);
    for (i = 0; i < n; ++i) {
        A[i][i] = 1 / p[i];
        for (j = i + 1; j < n; ++j) {
            sum = 0;
            for (k = i; k < j; ++k) {
                sum -= A[j][k] * A[k][i];
            }
            A[j][i] = sum / p[j];
        }
    }
    free(p);
}

void inverseMatrix_CholeskyMethod(int n, float** A) {
    unsigned i, j, k;
    choldcsl(n, A);
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            A[i][j] = 0.0;
        }
    }
    for (i = 0; i < n; i++) {
        A[i][i] *= A[i][i];
        for (k = i + 1; k < n; ++k) {
            A[i][i] += A[k][i] * A[k][i];
        }
        for (j = i + 1; j < n; ++j) {
            for (k = j; k < n; ++k) {
                A[i][j] += A[k][i] * A[k][j];
            }
        }
    }
    for (i = 0; i < n; ++i) {
        for (j = 0; j < i; ++j) {
            A[i][j] = A[j][i];
        }
    }
}

//Multiply matrix M transpose by M
void Mt_byM_multiply(int i, int j, float** M, float** Result) {
    float SUM;
    for (unsigned I = 0; I < j; ++I) {
        for (unsigned J = I; J < j; ++J) {
            SUM = 0.0f;
            for (unsigned K = 0; K < i; ++K) {
                //printf("%.3f %.3f\n", M[K][I], M[K][J]);
                SUM += M[K][I] * M[K][J];
            }
            Result[J][I] = SUM;
            Result[I][J] = SUM;
        }
    }
}

//Multiply matrix M by M transpose
void M_byMt_multiply(int i, int j, float** M, float** Result) {
    float SUM;
    for (unsigned I = 0; I < i; ++I) {
        for (unsigned J = 0; J < i; ++J) {
            SUM = 0.0;
            for (unsigned K = 0; K < j; ++K) {
                SUM += M[I][K] * M[J][K];
            }
            Result[I][J] = SUM;
        }
    }
}

void calculate_rmse(const mat_t& W_c, const mat_t& H_c, const char* srcdir, int k) {
    auto t1 = std::chrono::high_resolution_clock::now();
    int i, j;
    double v, rmse = 0;
    int num_insts = 0;
    int nans_count = 0;

    char meta_filename[1024];
    sprintf(meta_filename, "%s/meta", srcdir);
    FILE* fp = fopen(meta_filename, "r");
    if (fp == nullptr) {
        printf("Can't open meta input file.\n");
        exit(1);
    }

    char buf_train[1024], buf_test[1024], test_file_name[1024], train_file_name[1024];
    unsigned m, n, nnz, nnz_test;
    CHECK_FSCAN(fscanf(fp, "%u %u", &m, &n),2);
    CHECK_FSCAN(fscanf(fp, "%u %1023s", &nnz, buf_train),2);
    CHECK_FSCAN(fscanf(fp, "%u %1023s", &nnz_test, buf_test),2);
    sprintf(test_file_name, "%s/%s", srcdir, buf_test);
    sprintf(train_file_name, "%s/%s", srcdir, buf_train);
    fclose(fp);

    FILE* test_fp = fopen(test_file_name, "r");
    if (test_fp == nullptr) {
        printf("Can't open test file.\n");
        exit(1);
    }

    while (fscanf(test_fp, "%d %d %lf", &i, &j, &v) != EOF) {
        double pred_v = 0;
        for (int t = 0; t < k; t++) {
            pred_v += W_c[i - 1][t] * H_c[j - 1][t];
        }
        num_insts++;
        double tmp = (pred_v - v) * (pred_v - v);
        if (tmp == tmp) {
            rmse += tmp;
        } else {
            nans_count++;
        }
    }
    fclose(test_fp);

    double nans_percentage = (double) nans_count / (double) num_insts;
    printf("NaNs percentage: %lf, NaNs Count: %d, Total Insts: %d\n", nans_percentage, nans_count, num_insts);
    rmse = sqrt(rmse / num_insts);
    printf("CALCULATED RMSE = %lf.\n", rmse);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT12 = t2 - t1;
    std::cout << "Predict Time: " << deltaT12.count() << " s.\n";
}

void exit_with_help() {
    printf(
            "Usage: clMF [options] data_dir\n"
            "options:\n"
            "    -c : full path to the kernel code (default x)\n"
            "    -k rank : set the rank (default 10)\n"
            "    -l lambda : set the regularization parameter lambda (default 0.1)\n"
            "    -t max_iter: set the number of iterations (default 5)\n"
            "    -P platform_id: select a platform (default 0)\n"
            "    -nBlocks: Number of blocks on cuda (default 16)\n"
            "    -nThreadsPerBlock: Number of threads per block on cuda (default 32)\n"
    );
    exit(1);
}

parameter parse_command_line(int argc, char** argv, char* input_dir, char* kernel_code) {
    // default values have been set by the constructor
    parameter param;
    // parse options
    int i;
    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            break;
        }
        if (++i >= argc) {
            exit_with_help();
        }
        if (strcmp(argv[i - 1], "-nBlocks") == 0) {
            param.nBlocks = atoi(argv[i]);
        } else if (strcmp(argv[i - 1], "-nThreadsPerBlock") == 0) {
            param.nThreadsPerBlock = atoi(argv[i]);
        } else {
            switch (argv[i - 1][1]) {
                case 'c':
                    sprintf(kernel_code, "%s", argv[i]);
                    break;
                case 'k':
                    param.k = atoi(argv[i]);
                    break;
                case 'l':
                    param.lambda = atof(argv[i]);
                    break;
                case 't':
                    param.maxiter = atoi(argv[i]);
                    break;
                case 'P':
                    param.platform_id = atoi(argv[i]);
                    break;
                default:
                    fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
                    exit_with_help();
                    break;
            }
        }

    }

    if (i >= argc) {
        exit_with_help();
    }

    sprintf(input_dir, "%s", argv[i]);
    return param;
}


int main(int argc, char* argv[]) {
    auto t8 = std::chrono::high_resolution_clock::now();
    char device_type[4] = {'g', 'p', 'u', '\0'};
    const char* opencl_filename = "../kcode/ALS.cl";
    char srcdir[1024];

    smat_t R;
    parameter param = parse_command_line(argc, argv, srcdir, nullptr);
    mat_t W_c, H_c;

    cl_int status;
    cl_int err;
    cl_uint NumDevice;
    cl_platform_id platform;

    getPlatform(platform, param.platform_id);
    printf("[info] - the selected platform: %d, device type: %s\n", param.platform_id, device_type);
    cl_device_id* devices = getCl_device_id(platform, device_type);
    cl_context context = clCreateContext(nullptr, 1, devices, nullptr, nullptr, nullptr);
    CL_CHECK(clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &NumDevice, nullptr));
    printf("[info] - %d devices found!\n", NumDevice);
    cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, nullptr);

    printf("[info] - The kernel to be compiled: %s\n", opencl_filename);
    std::string sourceStr;
    status = convertToString(opencl_filename, sourceStr);
    if (status == -1) { exit(-1); }
    const char* source = sourceStr.c_str();
    size_t sourceSize[] = {strlen(source)};
    cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, nullptr);

    status = clBuildProgram(program, 1, devices, nullptr, nullptr, nullptr);

    size_t length;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
    char* buffer = (char*) malloc(length + 1);
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, length, buffer, nullptr);

    if (buffer != nullptr) {
        printf("[build info]:\n%s", buffer);
        free(buffer);
    }

    if (status != CL_SUCCESS) {
        std::cout << "ERROR:Could not compile OpenCl code !\n";
        exit(1);
    } else {
        std::cout << "[build info]: Compiled OpenCl code !\n";
    }

    puts("ALS-OpenCL-Parallel Programming: starts!");

    auto t3 = std::chrono::high_resolution_clock::now();
    bool with_weights = false;
    bool ifALS = true;
    load(srcdir, R, ifALS, with_weights);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT34 = t4 - t3;
    std::cout << "Load R Time: " << deltaT34.count() << " s.\n";

    initial_col(W_c, R.rows, param.k);
    initial_col(H_c, R.cols, param.k);

    int k = param.k;
    float lambda = param.lambda;
    long rows = R.rows;
    long cols = R.cols;
    int nBlocks = param.nBlocks;
    int nThreadsPerBlock = param.nThreadsPerBlock;
    int maxiter = param.maxiter;
    long* col_ptr = R.col_ptr, * row_ptr = R.row_ptr;
    unsigned* row_idx = R.row_idx, * col_idx = R.col_idx;
    unsigned* colMajored_sparse_idx = R.colMajored_sparse_idx;
    float* val = R.val;

    float* submatrix;
    submatrix = (float*) malloc(k * k * sizeof(float));
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            submatrix[i * k + j] = 0.0f;
        }
    }

    float* W, * H;
    W = (float*) malloc(k * R.rows * sizeof(float));
    H = (float*) malloc(k * R.cols * sizeof(float));

    size_t nbits_W_ = R.rows * k * sizeof(float);
    size_t nbits_H_ = R.cols * k * sizeof(float);

    int indexPosition = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < k; ++j) {
            W[indexPosition] = 0.0;
            ++indexPosition;
        }
    }

    int indexPosition1 = 0;
    for (int i = 0; i < R.cols; ++i) {
        for (int j = 0; j < k; ++j) {
            H[indexPosition1] = H_c[i][j];
            ++indexPosition1;
        }
    }

    cl_mem row_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_row_ptr, (void*) row_ptr, nullptr);
    cl_mem col_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_col_idx, (void*) col_idx, nullptr);
    cl_mem col_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_col_ptr, (void*) col_ptr, nullptr);
    cl_mem row_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_row_idx, (void*) row_idx, nullptr);
    cl_mem colMajored_sparse_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_colMajored_sparse_idx, (void*) colMajored_sparse_idx, nullptr);
    cl_mem valBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_val, (void*) val, nullptr);
    cl_mem WBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_W_, (void*) W, nullptr);
    cl_mem HBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_H_, (void*) H, nullptr);
    cl_mem pBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * sizeof(float), nullptr, nullptr);
    cl_mem subVecBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * sizeof(float), nullptr, nullptr);
    cl_mem subMatBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * k * sizeof(float), nullptr, nullptr);
    cl_mem subMatrixBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, k * k * sizeof(float), (void*) submatrix, nullptr);
    cl_mem p_Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * sizeof(float), nullptr, nullptr);
    cl_mem subVec_Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * sizeof(float), nullptr, nullptr);
    cl_mem subMat_Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * k * sizeof(float), nullptr, nullptr);

    cl_kernel updateWOverH_kernel = clCreateKernel(program, "updateW_overH_kernel", &err);
    CHECK_ERROR(err);
    cl_kernel updateHOverW_kernel = clCreateKernel(program, "updateH_overW_kernel", &err);
    CHECK_ERROR(err);

    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 0, sizeof(long), &rows));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 1, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 2, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 3, sizeof(cl_mem), (void*) &colMajored_sparse_idxBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 4, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 5, sizeof(float), &lambda));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 6, sizeof(int), &k));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 7, sizeof(cl_mem), (void*) &WBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 8, sizeof(cl_mem), (void*) &HBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 9, sizeof(cl_mem), (void*) &pBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 10, sizeof(cl_mem), (void*) &subVecBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 11, sizeof(cl_mem), (void*) &subMatBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 12, sizeof(cl_mem), (void*) &subMatrixBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 0, sizeof(long), &cols));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 4, sizeof(float), &lambda));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 5, sizeof(int), &k));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 6, sizeof(cl_mem), (void*) &WBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 7, sizeof(cl_mem), (void*) &HBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 8, sizeof(cl_mem), (void*) &p_Buffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 9, sizeof(cl_mem), (void*) &subVec_Buffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 10, sizeof(cl_mem), (void*) &subMat_Buffer));

    auto t1 = std::chrono::high_resolution_clock::now();
    for (unsigned int ite = 0; ite < maxiter; ite++) {
        size_t global_work_size[1] = {static_cast<size_t>(nBlocks * nThreadsPerBlock)};
        size_t local_work_size[1] = {static_cast<size_t>(nThreadsPerBlock)};

        /** update_W_Over_H */
        cl_event enentPoint;
        CL_CHECK(clEnqueueNDRangeKernel(commandQueue, updateWOverH_kernel, 1, nullptr, global_work_size, local_work_size, 0, nullptr, &enentPoint));
        clWaitForEvents(1, &enentPoint);
        clReleaseEvent(enentPoint);
/*
        status=clEnqueueReadBuffer(commandQueue, WBuffer, CL_TRUE, 0, nbits_W_, W, 0, NULL, NULL);
        status=clEnqueueReadBuffer(commandQueue, HBuffer, CL_TRUE, 0, nbits_H_, H, 0, NULL, NULL);
        status=clEnqueueReadBuffer(commandQueue, subMatrixBuffer, CL_TRUE, 0, k*k*sizeof(float), submatrix, 0, NULL, NULL);
        std::cout<<"update_W_over_H   W:\n";
        for(int df=0;df<10;df++)
        {
            for(int fd=0;fd<5;fd++)
            {
                std::cout<<W[df*k+fd]<<"  ";
            }
            std::cout<<"\n";
        }

        std::cout<<"update_W_over_H   H:\n";
        for(int df=0;df<10;df++)
        {
            for(int fd=0;fd<5;fd++)
            {
                std::cout<<H[df*k+fd]<<"  ";
            }
            std::cout<<"\n";
        }

        std::cout<<"update_W_over_H   subMatrix:\n";
        for(int df=0;df<10;df++)
        {
            for(int fd=0;fd<10;fd++)
            {
                std::cout<<submatrix[df*k+fd]<<"  ";
            }
            std::cout<<"\n";
        }
*/
        /** update_H_Over_W */
        cl_event enentPoint1;
        CL_CHECK(clEnqueueNDRangeKernel(commandQueue, updateHOverW_kernel, 1, nullptr, global_work_size, local_work_size, 0, nullptr, &enentPoint1));
        clWaitForEvents(1, &enentPoint1);
        clReleaseEvent(enentPoint1);
/*
        printf("ddd.\n");
        status=clEnqueueReadBuffer(commandQueue, WBuffer, CL_TRUE, 0, nbits_W_, W, 0, NULL, NULL);
        status=clEnqueueReadBuffer(commandQueue, HBuffer, CL_TRUE, 0, nbits_H_, H, 0, NULL, NULL);
        std::cout<<"update_H_over_W   W:\n";
        for(int df=0;df<10;df++)
        {
            for(int fd=0;fd<5;fd++)
            {
                std::cout<<W[df*k+fd]<<"  ";
            }
            std::cout<<"\n";
        }
        std::cout<<"update_H_over_W   H:\n";
        for(int df=0;df<10;df++)
        {
            for(int fd=0;fd<5;fd++)
            {
                std::cout<<H[df*k+fd]<<"  ";
            }
            std::cout<<"\n";
        }
*/
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT = t2 - t1;
    std::cout << "Training Time: " << deltaT.count() << " s.\n";

    CL_CHECK(clEnqueueReadBuffer(commandQueue, WBuffer, CL_TRUE, 0, nbits_W_, W, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(commandQueue, HBuffer, CL_TRUE, 0, nbits_H_, H, 0, nullptr, nullptr));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < k; ++j) {
            W_c[i][j] = W[i * k + j];
        }
    }
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < k; ++j) {
            H_c[i][j] = H[i * k + j];
        }
    }

    calculate_rmse(W_c, H_c, srcdir, k);

    CL_CHECK(clReleaseKernel(updateHOverW_kernel));
    CL_CHECK(clReleaseKernel(updateWOverH_kernel));
    CL_CHECK(clReleaseProgram(program));
    CL_CHECK(clReleaseMemObject(row_ptrBuffer));
    CL_CHECK(clReleaseMemObject(col_idxBuffer));
    CL_CHECK(clReleaseMemObject(col_ptrBuffer));
    CL_CHECK(clReleaseMemObject(row_idxBuffer));
    CL_CHECK(clReleaseMemObject(colMajored_sparse_idxBuffer));
    CL_CHECK(clReleaseMemObject(valBuffer));
    CL_CHECK(clReleaseMemObject(WBuffer));
    CL_CHECK(clReleaseMemObject(HBuffer));
    CL_CHECK(clReleaseMemObject(subMatrixBuffer));
    CL_CHECK(clReleaseCommandQueue(commandQueue));
    CL_CHECK(clReleaseContext(context));
    free(devices);

    auto t9 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT89 = t9 - t8;
    std::cout << "total Time: " << deltaT89.count() << " s.\n";
    return 0;
}
