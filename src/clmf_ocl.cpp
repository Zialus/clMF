#include "tools.h"

extern std::chrono::duration<double> deltaT12;
extern std::chrono::duration<double> deltaTAB;

void clmf(SparseMatrix& R, MatData& W_c, MatData& H_c, TestData &T, parameter& param, char filename[]) {
    auto tA = std::chrono::high_resolution_clock::now();

    cl_int status;
    cl_int err;
    cl_platform_id platform = getPlatform(param.platform_id);
    cl_device_id* devices = getDevices(platform, param.device_type);
    report_device(devices[0]);
    cl_context context = clCreateContext(nullptr, 1, devices, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_uint NumDevice;
    CL_CHECK(clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &NumDevice, nullptr));
    assert(NumDevice == 1);
    cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);
    printf("[INFO] Connected!\n");

    printf("[INFO] - The kernel to be compiled: %s\n", filename);
    std::string sourceStr;
    convertToString(filename, sourceStr);
    const char* source = sourceStr.c_str();
    size_t sourceSize[] = {strlen(source)};
    cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, &err);
    CL_CHECK(err);

    char options[1024];
    snprintf(options, sizeof(options), "-DVALUE_TYPE=%s -DK_SIZE=%u", getT(sizeof(VALUE_TYPE)), param.k);
    status = clBuildProgram(program, 1, devices, options, nullptr, nullptr);

    size_t length;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
    char* buffer = (char*) malloc(length + 1);
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, length, buffer, nullptr);

    if (buffer != nullptr && strcmp(buffer, "") != 0 && strcmp(buffer, "\n") != 0) {
        printf("[OpenCL Compiler INFO]:\n%s\n", buffer);
        free(buffer);
    } else {
        printf("[OpenCL Compiler]: No info to print\n");
    }

    CL_CHECK(status);
    std::cout << "[INFO]: Compiled OpenCl code successfully!\n";

    CL_CHECK(clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, nullptr, &length));
    char* buffer2 = (char*) malloc(length + 1);
    CL_CHECK(clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, length, buffer2, nullptr));
    if (buffer2 != nullptr && param.verbose) {
        printf("[Kernels]: %s\n", buffer2);
        free(buffer2);
    }

    auto tB = std::chrono::high_resolution_clock::now();
    deltaTAB = tB - tA;
    std::cout << "[INFO] Initiating OpenCL Time: " << deltaTAB.count() << " s.\n";

    unsigned k = param.k;
    int nBlocks = param.nBlocks;
    int nThreadsPerBlock = param.nThreadsPerBlock;

    VALUE_TYPE* submatrix = (VALUE_TYPE*) malloc(k * k * sizeof(VALUE_TYPE));
    for (unsigned i = 0; i < k; i++) {
        for (unsigned j = 0; j < k; j++) {
            submatrix[i * k + j] = 0.0;
        }
    }

    VALUE_TYPE* W = (VALUE_TYPE*) malloc(k * R.rows * sizeof(VALUE_TYPE));
    for (long i = 0; i < R.rows; ++i) {
        for (unsigned j = 0; j < k; ++j) {
            W[i * k + j] = 0.0;
        }
    }

    VALUE_TYPE* H = (VALUE_TYPE*) malloc(k * R.cols * sizeof(VALUE_TYPE));
    for (long i = 0; i < R.cols; ++i) {
        for (unsigned j = 0; j < k; ++j) {
            H[i * k + j] = H_c[i][j];
        }
    }

    size_t nbits_W_ = R.rows * k * sizeof(VALUE_TYPE);
    size_t nbits_H_ = R.cols * k * sizeof(VALUE_TYPE);

    size_t nbits_col_ptr = (R.cols + 1) * sizeof(unsigned);
    size_t nbits_row_ptr = (R.rows + 1) * sizeof(unsigned);

    size_t nbits_idx = R.nnz * sizeof(unsigned);

    size_t nbits_val = R.nnz * sizeof(VALUE_TYPE);

    // creating buffers
    cl_mem row_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_row_ptr, R.get_csr_row_ptr(), &err);
    CL_CHECK(err);
    cl_mem col_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_idx, R.get_csr_col_indx(), &err);
    CL_CHECK(err);
    cl_mem col_ptrBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_col_ptr, R.get_csc_col_ptr(), &err);
    CL_CHECK(err);
    cl_mem row_idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_idx, R.get_csc_row_indx(), &err);
    CL_CHECK(err);
    cl_mem val_tBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_val, R.get_csr_val(), &err);
    CL_CHECK(err);
    cl_mem valBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_val, R.get_csc_val(), &err);
    CL_CHECK(err);
    cl_mem WBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_W_, W, &err);
    CL_CHECK(err);
    cl_mem HBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbits_H_, H, &err);
    CL_CHECK(err);
    cl_mem pBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * sizeof(VALUE_TYPE), nullptr, &err);
    CL_CHECK(err);
    cl_mem subVecBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * sizeof(VALUE_TYPE), nullptr, &err);
    CL_CHECK(err);
    cl_mem subMatBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * k * sizeof(VALUE_TYPE), nullptr, &err);
    CL_CHECK(err);
    cl_mem subMatrixBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, k * k * sizeof(VALUE_TYPE), submatrix, &err);
    CL_CHECK(err);
    cl_mem p_Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * sizeof(VALUE_TYPE), nullptr, &err);
    CL_CHECK(err);
    cl_mem subVec_Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * sizeof(VALUE_TYPE), nullptr, &err);
    CL_CHECK(err);
    cl_mem subMat_Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nBlocks * nThreadsPerBlock * k * k * sizeof(VALUE_TYPE), nullptr, &err);
    CL_CHECK(err);

    // RMSE related buffers
    cl_mem test_rowBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, T.nnz * sizeof(unsigned), T.getTestRow(), &err);
    CL_CHECK(err);
    cl_mem test_colBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, T.nnz * sizeof(unsigned), T.getTestCol(), &err);
    CL_CHECK(err);
    cl_mem test_valBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, T.nnz * sizeof(unsigned), T.getTestVal(), &err);
    CL_CHECK(err);
    cl_mem pred_vBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, T.nnz * sizeof(VALUE_TYPE), nullptr, &err);
    CL_CHECK(err);
    cl_mem rmseBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, T.nnz * sizeof(VALUE_TYPE), nullptr, &err);
    CL_CHECK(err);
    cl_mem emptyBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, T.nnz * sizeof(VALUE_TYPE), nullptr, &err);
    CL_CHECK(err);

    // creating and building kernels
    cl_kernel updateWOverH_kernel = clCreateKernel(program, "updateW_overH_kernel", &err);
    CL_CHECK(err);
    cl_kernel updateHOverW_kernel = clCreateKernel(program, "updateH_overW_kernel", &err);
    CL_CHECK(err);
    cl_kernel gpuRMSE_kernel = clCreateKernel(program, "GPU_rmse", &err);
    CL_CHECK(err);

    // setting kernel arguments
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 0, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 1, sizeof(cl_mem), &row_ptrBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 2, sizeof(cl_mem), &col_idxBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 3, sizeof(cl_mem), &val_tBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 4, sizeof(cl_mem), &valBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 5, sizeof(VALUE_TYPE), &param.lambda));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 6, sizeof(unsigned), &k));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 7, sizeof(cl_mem), &WBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 8, sizeof(cl_mem), &HBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 9, sizeof(cl_mem), &pBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 10, sizeof(cl_mem), &subVecBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 11, sizeof(cl_mem), &subMatBuffer));
    CL_CHECK(clSetKernelArg(updateWOverH_kernel, 12, sizeof(cl_mem), &subMatrixBuffer));

    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 0, sizeof(unsigned), &R.cols));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 1, sizeof(cl_mem), &col_ptrBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 2, sizeof(cl_mem), &row_idxBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 3, sizeof(cl_mem), &valBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 4, sizeof(VALUE_TYPE), &param.lambda));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 5, sizeof(unsigned), &k));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 6, sizeof(cl_mem), &WBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 7, sizeof(cl_mem), &HBuffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 8, sizeof(cl_mem), &p_Buffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 9, sizeof(cl_mem), &subVec_Buffer));
    CL_CHECK(clSetKernelArg(updateHOverW_kernel, 10, sizeof(cl_mem), &subMat_Buffer));

    bool isALS = true;

    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 0, sizeof(cl_mem), &test_rowBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 1, sizeof(cl_mem), &test_colBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 2, sizeof(cl_mem), &test_valBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 3, sizeof(cl_mem), &pred_vBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 4, sizeof(cl_mem), &rmseBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 5, sizeof(cl_mem), &WBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 6, sizeof(cl_mem), &HBuffer));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 7, sizeof(unsigned), &T.nnz));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 8, sizeof(unsigned), &k));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 9, sizeof(unsigned), &R.rows));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 10, sizeof(unsigned), &R.cols));
    CL_CHECK(clSetKernelArg(gpuRMSE_kernel, 11, sizeof(unsigned), &isALS));

    size_t global_work_size[1] = {static_cast<size_t>(param.nBlocks * param.nThreadsPerBlock)};
    size_t local_work_size[1] = {static_cast<size_t>(param.nThreadsPerBlock)};
    printf("[INFO] - blocks: %d | threads per block: %d | global_work_size: %zu | local_work_size: %zu !\n",
           param.nBlocks, param.nThreadsPerBlock, global_work_size[0], local_work_size[0]);

    if (param.verbose) {
        size_t local;
        CL_CHECK(clGetKernelWorkGroupInfo(updateHOverW_kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr));
        printf("[VERBOSE] local_work_size for updateHOverW_kernel should be: %zu\n",local);
        CL_CHECK(clGetKernelWorkGroupInfo(updateWOverH_kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr));
        printf("[VERBOSE] local_work_size for updateWOverH_kernel should be: %zu\n",local);
        CL_CHECK(clGetKernelWorkGroupInfo(gpuRMSE_kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr));
        printf("[VERBOSE] local_work_size for gpuRMSE_kernel should be: %zu\n",local);
    }

    double t_update_ratings_acc = 0;

    float* rmseVec = (float*) malloc((T.nnz) * sizeof(float));

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "[INFO] Computing clMF OpenCL..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int ite = 1; ite <= param.maxiter; ite++) {

        double t_update_ratings = 0;

        /** update_W_Over_H */
        cl_event eventPoint0;
        CL_CHECK(clEnqueueNDRangeKernel(commandQueue, updateWOverH_kernel, 1, nullptr, global_work_size, local_work_size, 0, nullptr, &eventPoint0));
        CL_CHECK(clWaitForEvents(1, &eventPoint0));

        t_update_ratings += executionTime(eventPoint0);

        CL_CHECK(clReleaseEvent(eventPoint0));
/*
        CL_CHECK(clEnqueueReadBuffer(commandQueue, WBuffer, CL_TRUE, 0, nbits_W_, W, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(commandQueue, HBuffer, CL_TRUE, 0, nbits_H_, H, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(commandQueue, subMatrixBuffer, CL_TRUE, 0, k*k*sizeof(VALUE_TYPE), submatrix, 0, nullptr, nullptr));
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
        cl_event eventPoint1;
        CL_CHECK(clEnqueueNDRangeKernel(commandQueue, updateHOverW_kernel, 1, nullptr, global_work_size, local_work_size, 0, nullptr, &eventPoint1));
        CL_CHECK(clWaitForEvents(1, &eventPoint1));

        t_update_ratings += executionTime(eventPoint1);

        CL_CHECK(clReleaseEvent(eventPoint1));
/*
        printf("ddd.\n");
        CL_CHECK(clEnqueueReadBuffer(commandQueue, WBuffer, CL_TRUE, 0, nbits_W_, W, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(commandQueue, HBuffer, CL_TRUE, 0, nbits_H_, H, 0, nullptr, nullptr));
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
        t_update_ratings_acc += t_update_ratings;

        /** Calculate RMSE*/
        cl_event eventPoint3;

        CL_CHECK(clEnqueueCopyBuffer(commandQueue, emptyBuffer, rmseBuffer, 0, 0, (T.nnz) * sizeof(float), 0, nullptr, &eventPoint3));
        CL_CHECK(clEnqueueCopyBuffer(commandQueue, emptyBuffer, pred_vBuffer, 0, 0, (T.nnz) * sizeof(float), 0, nullptr, &eventPoint3));

//        size_t gws_rmse[1] = {((T.nnz + 1023) / 1024) * 1024};
//        size_t lws_rmse[1] = {1024};
//        CL_CHECK(clEnqueueNDRangeKernel(commandQueue, gpuRMSE_kernel, 1, nullptr, gws_rmse, lws_rmse, 0, nullptr, &eventPoint3));
        CL_CHECK(clEnqueueNDRangeKernel(commandQueue, gpuRMSE_kernel, 1, nullptr, global_work_size, local_work_size, 0, nullptr, &eventPoint3));

        CL_CHECK(clWaitForEvents(1, &eventPoint3));

        double rmse_time = executionTime(eventPoint3);

        CL_CHECK(clReleaseEvent(eventPoint3));

        CL_CHECK(clEnqueueReadBuffer(commandQueue, rmseBuffer, CL_TRUE, 0, (T.nnz) * sizeof(float), rmseVec, 0, nullptr, nullptr));

        double tot_rmse = 0;
        double f_rmse = 0;

        for (unsigned i = 0; i < T.nnz; ++i) {
            tot_rmse += rmseVec[i];
        }
        f_rmse = sqrt(tot_rmse / T.nnz);

        printf("[-INFO-] iteration num %d \tupdate_time %.4lf|%.4lfs \tRMSE=%f time:%fs\n",
                ite, t_update_ratings, t_update_ratings_acc, f_rmse, rmse_time);

    }
    auto t2 = std::chrono::high_resolution_clock::now();
    deltaT12 = t2 - t1;
    std::cout << "[INFO] OCL Training Time: " << deltaT12.count() << " s.\n";

    CL_CHECK(clEnqueueReadBuffer(commandQueue, WBuffer, CL_TRUE, 0, nbits_W_, W, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(commandQueue, HBuffer, CL_TRUE, 0, nbits_H_, H, 0, nullptr, nullptr));

    CL_CHECK(clReleaseKernel(updateHOverW_kernel));
    CL_CHECK(clReleaseKernel(updateWOverH_kernel));
    CL_CHECK(clReleaseKernel(gpuRMSE_kernel));
    CL_CHECK(clReleaseProgram(program));
    CL_CHECK(clReleaseMemObject(row_ptrBuffer));
    CL_CHECK(clReleaseMemObject(col_idxBuffer));
    CL_CHECK(clReleaseMemObject(col_ptrBuffer));
    CL_CHECK(clReleaseMemObject(row_idxBuffer));
    CL_CHECK(clReleaseMemObject(val_tBuffer));
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
    CL_CHECK(clReleaseMemObject(test_rowBuffer));
    CL_CHECK(clReleaseMemObject(test_colBuffer));
    CL_CHECK(clReleaseMemObject(test_valBuffer));
    CL_CHECK(clReleaseMemObject(pred_vBuffer));
    CL_CHECK(clReleaseMemObject(rmseBuffer));
    CL_CHECK(clReleaseMemObject(emptyBuffer));
    CL_CHECK(clReleaseCommandQueue(commandQueue));
    CL_CHECK(clReleaseContext(context));
    CL_CHECK(clReleaseDevice(devices[0]));
    free(devices);
    free(rmseVec);

    for (long i = 0; i < R.rows; ++i) {
        for (unsigned j = 0; j < k; ++j) {
            W_c[i][j] = W[i * k + j];
        }
    }
    for (long i = 0; i < R.cols; ++i) {
        for (unsigned j = 0; j < k; ++j) {
            H_c[i][j] = H[i * k + j];
        }
    }

}
