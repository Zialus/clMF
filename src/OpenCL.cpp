#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "tools.h"

void calculate_rmse(const mat_t& W_c, const mat_t& H_c, const char* srcdir, int k) {
    char meta_filename[1024];
    sprintf(meta_filename, "%s/meta", srcdir);
    FILE* fp = fopen(meta_filename, "r");
    if (fp == nullptr) {
        printf("Can't open meta input file.\n");
        exit(EXIT_FAILURE);
    }

    char buf_train[1024], buf_test[1024], test_file_name[2048], train_file_name[2048];
    unsigned m, n, nnz, nnz_test;
    CHECK_FSCAN(fscanf(fp, "%u %u", &m, &n), 2);
    CHECK_FSCAN(fscanf(fp, "%u %1023s", &nnz, buf_train), 2);
    CHECK_FSCAN(fscanf(fp, "%u %1023s", &nnz_test, buf_test), 2);
    snprintf(test_file_name, sizeof(test_file_name), "%s/%s", srcdir, buf_test);
    snprintf(train_file_name, sizeof(train_file_name), "%s/%s", srcdir, buf_train);
    fclose(fp);

    FILE* test_fp = fopen(test_file_name, "r");
    if (test_fp == nullptr) {
        printf("Can't open test file.\n");
        exit(EXIT_FAILURE);
    }

    double rmse = 0;
    int num_insts = 0;
    int nans_count = 0;

    int i, j;
    double v;

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
}

int main(int argc, char* argv[]) {
    auto t7 = std::chrono::high_resolution_clock::now();

    parameter param = parse_command_line(argc, argv);

    if (param.verbose) {
        print_all_the_platforms();
        print_all_the_info();
    }

    auto tA = std::chrono::high_resolution_clock::now();
    cl_int status;
    cl_int err;
    cl_uint NumDevice;
    cl_platform_id platform;
    getPlatform(platform, param.platform_id);
    cl_device_id* devices = getDevice(platform, param.device_type);
    report_device(devices[0]);
    cl_context context = clCreateContext(nullptr, 1, devices, nullptr, nullptr, &err);
    CL_CHECK(err);
    CL_CHECK(clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &NumDevice, nullptr));
    assert(NumDevice == 1);
    cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, nullptr);

    printf("[info] - The kernel to be compiled: %s\n", param.opencl_filename);
    std::string sourceStr;
    convertToString(param.opencl_filename, sourceStr);
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

    CL_CHECK(status);
    std::cout << "[build info]:Compiled OpenCl code!\n";


    auto tB = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaTAB = tB - tA;
    std::cout << "Initiating OpenCL Time: " << deltaTAB.count() << " s.\n";

    auto t3 = std::chrono::high_resolution_clock::now();
    smat_t R;
    bool with_weights = false;
    bool ifALS = true;
    std::cout << "[info]Loading R...\n";
    load(param.src_dir, R, ifALS, with_weights);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT34 = t4 - t3;
    std::cout << "Load R Time: " << deltaT34.count() << " s.\n";

    int k = param.k;
    int nBlocks = param.nBlocks;
    int nThreadsPerBlock = param.nThreadsPerBlock;

    mat_t W_c;
    mat_t H_c;
    initial_col(W_c, R.rows, param.k);
    initial_col(H_c, R.cols, param.k);

    float* submatrix = (float*) malloc(k * k * sizeof(float));
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            submatrix[i * k + j] = 0.0f;
        }
    }

    float* W = (float*) malloc(k * R.rows * sizeof(float));
    float* H = (float*) malloc(k * R.cols * sizeof(float));
    for (int i = 0; i < R.rows; ++i) {
        for (int j = 0; j < k; ++j) {
            W[i * k + j] = 0.0;
        }
    }
    for (int i = 0; i < R.cols; ++i) {
        for (int j = 0; j < k; ++j) {
            H[i * k + j] = H_c[i][j];
        }
    }

    size_t nbits_W_ = R.rows * k * sizeof(float);
    size_t nbits_H_ = R.cols * k * sizeof(float);

    // creating buffers
    cl_mem row_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_row_ptr, (void*) R.row_ptr, &err);
    CL_CHECK(err);
    cl_mem col_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_col_idx, (void*) R.col_idx, &err);
    CL_CHECK(err);
    cl_mem col_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_col_ptr, (void*) R.col_ptr, &err);
    CL_CHECK(err);
    cl_mem row_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_row_idx, (void*) R.row_idx, &err);
    CL_CHECK(err);
    cl_mem colMajored_sparse_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_colMajored_sparse_idx, (void*) R.colMajored_sparse_idx, &err);
    CL_CHECK(err);
    cl_mem valBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, R.nbits_val, (void*) R.val, &err);
    CL_CHECK(err);
    cl_mem WBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_W_, (void*) W, &err);
    CL_CHECK(err);
    cl_mem HBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_H_, (void*) H, &err);
    CL_CHECK(err);
    cl_mem pBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem subVecBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem subMatBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * k * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem subMatrixBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, k * k * sizeof(float), (void*) submatrix, &err);
    CL_CHECK(err);
    cl_mem p_Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem subVec_Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem subMat_Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * k * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    // creating and building kernels
    cl_kernel updateWOverH_kernel = clCreateKernel(program, "updateW_overH_kernel", &err);
    CL_CHECK(err);
    cl_kernel updateHOverW_kernel = clCreateKernel(program, "updateH_overW_kernel", &err);
    CL_CHECK(err);

    // setting kernel arguments
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 0, sizeof(int), &R.rows));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 1, sizeof(cl_mem), (void*) &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 2, sizeof(cl_mem), (void*) &col_idxBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 3, sizeof(cl_mem), (void*) &colMajored_sparse_idxBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 4, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 5, sizeof(float), &param.lambda));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 6, sizeof(int), &k));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 7, sizeof(cl_mem), (void*) &WBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 8, sizeof(cl_mem), (void*) &HBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 9, sizeof(cl_mem), (void*) &pBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 10, sizeof(cl_mem), (void*) &subVecBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 11, sizeof(cl_mem), (void*) &subMatBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 12, sizeof(cl_mem), (void*) &subMatrixBuffer));

    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 0, sizeof(int), &R.cols));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 1, sizeof(cl_mem), (void*) &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 2, sizeof(cl_mem), (void*) &row_idxBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 3, sizeof(cl_mem), (void*) &valBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 4, sizeof(float), &param.lambda));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 5, sizeof(int), &k));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 6, sizeof(cl_mem), (void*) &WBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 7, sizeof(cl_mem), (void*) &HBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 8, sizeof(cl_mem), (void*) &p_Buffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 9, sizeof(cl_mem), (void*) &subVec_Buffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 10, sizeof(cl_mem), (void*) &subMat_Buffer));

    printf("[info] - threads per block: %d\n", param.nThreadsPerBlock);

    size_t local;
    CL_CHECK(clGetKernelWorkGroupInfo(updateHOverW_kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL));
    printf("local_work_size for updateHOverW_kernel should be: %zu\n",local);
    CL_CHECK(clGetKernelWorkGroupInfo(updateWOverH_kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL));
    printf("local_work_size for updateWOverH_kernel should be: %zu\n",local);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int ite = 0; ite < param.maxiter; ite++) {
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
    std::chrono::duration<double> deltaT12 = t2 - t1;
    std::cout << "Training Time: " << deltaT12.count() << " s.\n";

    CL_CHECK(clEnqueueReadBuffer(commandQueue, WBuffer, CL_TRUE, 0, nbits_W_, W, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(commandQueue, HBuffer, CL_TRUE, 0, nbits_H_, H, 0, nullptr, nullptr));

    for (int i = 0; i < R.rows; ++i) {
        for (int j = 0; j < k; ++j) {
            W_c[i][j] = W[i * k + j];
        }
    }
    for (int i = 0; i < R.cols; ++i) {
        for (int j = 0; j < k; ++j) {
            H_c[i][j] = H[i * k + j];
        }
    }

    auto t5 = std::chrono::high_resolution_clock::now();
    calculate_rmse(W_c, H_c, param.src_dir, k);
    auto t6 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT56 = t6 - t5;
    std::cout << "Predict Time: " << deltaT56.count() << " s.\n";

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
    CL_CHECK(clReleaseMemObject(pBuffer));
    CL_CHECK(clReleaseMemObject(subVecBuffer));
    CL_CHECK(clReleaseMemObject(subMatBuffer));
    CL_CHECK(clReleaseMemObject(subMatrixBuffer));
    CL_CHECK(clReleaseMemObject(p_Buffer));
    CL_CHECK(clReleaseMemObject(subVec_Buffer));
    CL_CHECK(clReleaseMemObject(subMat_Buffer));
    CL_CHECK(clReleaseCommandQueue(commandQueue));
    CL_CHECK(clReleaseContext(context));
    CL_CHECK(clReleaseDevice(devices[0]));
    free(devices);

    auto t8 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT78 = t8 - t7;
    std::cout << "Total Time: " << deltaT78.count() << " Parcial Sums:"
              << deltaT12.count() + deltaT34.count() + deltaT56.count() + deltaTAB.count() << " s.\n";
    return 0;
}
