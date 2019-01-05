#include <chrono>

#include "tools.h"
#include "extra.h"

std::chrono::duration<double> deltaT12;
std::chrono::duration<double> deltaTAB;

void doit(smat_t& R, mat_t& W_c, mat_t& H_c, parameter& param, char filename[]) {
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
    cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, nullptr);
    printf("[info] Connected!\n");

    printf("[info] - The kernel to be compiled: %s\n", filename);
    std::string sourceStr;
    convertToString(filename, sourceStr);
    const char* source = sourceStr.c_str();
    size_t sourceSize[] = {strlen(source)};
    cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, nullptr);

    char options[1024];
    snprintf(options, sizeof(options), " ");
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

    clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, nullptr, &length);
    char* buffer2 = (char*) malloc(length + 1);
    clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, length, buffer2, nullptr);
    if (buffer2 != nullptr) {
        printf("[Kernels]: %s\n", buffer2);
        free(buffer2);
    }

    auto tB = std::chrono::high_resolution_clock::now();
    deltaTAB = tB - tA;
    std::cout << "[INFO] Initiating OpenCL Time: " << deltaTAB.count() << " s.\n";

    unsigned k = param.k;
    int nBlocks = param.nBlocks;
    int nThreadsPerBlock = param.nThreadsPerBlock;

    float* submatrix = (float*) malloc(k * k * sizeof(float));
    for (unsigned i = 0; i < k; i++) {
        for (unsigned j = 0; j < k; j++) {
            submatrix[i * k + j] = 0.0f;
        }
    }

    float* W = (float*) malloc(k * R.rows * sizeof(float));
    for (unsigned i = 0; i < R.rows; ++i) {
        for (unsigned j = 0; j < k; ++j) {
            W[i * k + j] = 0.0;
        }
    }

    float* H = (float*) malloc(k * R.cols * sizeof(float));
    for (unsigned i = 0; i < R.cols; ++i) {
        for (unsigned j = 0; j < k; ++j) {
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

    size_t global_work_size[1] = {static_cast<size_t>(nBlocks * nThreadsPerBlock)};
    size_t local_work_size[1] = {static_cast<size_t>(nThreadsPerBlock)};
    printf("[info] - blocks: %d | threads per block: %d | global_work_size: %zu | local_work_size: %zu !\n",
           param.nBlocks, param.nThreadsPerBlock, global_work_size[0], local_work_size[0]);

    size_t local;
    CL_CHECK(clGetKernelWorkGroupInfo(updateHOverW_kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL));
    printf("[VERBOSE] local_work_size for updateHOverW_kernel should be: %zu\n",local);
    CL_CHECK(clGetKernelWorkGroupInfo(updateWOverH_kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL));
    printf("[VERBOSE] local_work_size for updateWOverH_kernel should be: %zu\n",local);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int ite = 0; ite < param.maxiter; ite++) {
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
    deltaT12 = t2 - t1;
    std::cout << "[info] Training Time: " << deltaT12.count() << " s.\n";

    CL_CHECK(clEnqueueReadBuffer(commandQueue, WBuffer, CL_TRUE, 0, nbits_W_, W, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(commandQueue, HBuffer, CL_TRUE, 0, nbits_H_, H, 0, nullptr, nullptr));

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

    for (unsigned i = 0; i < R.rows; ++i) {
        for (unsigned j = 0; j < k; ++j) {
            W_c[i][j] = W[i * k + j];
        }
    }
    for (unsigned i = 0; i < R.cols; ++i) {
        for (unsigned j = 0; j < k; ++j) {
            H_c[i][j] = H[i * k + j];
        }
    }

}

int main(int argc, char* argv[]) {
    auto t7 = std::chrono::high_resolution_clock::now();

    parameter param = parse_command_line(argc, argv);

    if (param.verbose) {
        print_all_the_info();
    }

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "[info] Loading R matrix..." << std::endl;
    auto t3 = std::chrono::high_resolution_clock::now();
    smat_t R;
    bool with_weights = false;
    bool ifALS = true;
    load(param.src_dir, R, ifALS, with_weights);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT34 = t4 - t3;
    std::cout << "[INFO] Loading rating data time: " << deltaT34.count() << "s.\n";
    std::cout << "------------------------------------------------------" << std::endl;

    mat_t W_c;
    mat_t H_c;
    initial_col(W_c, R.rows, param.k);
    initial_col(H_c, R.cols, param.k);

    mat_t W_ref;
    mat_t H_ref;
    initial_col(W_ref, R.rows, param.k);
    initial_col(H_ref, R.cols, param.k);

    doit(R, W_c, H_c, param, param.opencl_filename);

    std::chrono::duration<double> deltaT56{};
    std::chrono::duration<double> deltaT9_10{};

    // Predict RMSE with the W and H matrices produced by OpenCL kernels
    if (param.do_predict == 1) {
        std::cout << "------------------------------------------------------" << std::endl;
        auto t5 = std::chrono::high_resolution_clock::now();
        calculate_rmse(W_c, H_c, param.src_dir, param.k);
        auto t6 = std::chrono::high_resolution_clock::now();
        deltaT56 = t6 - t5;
        std::cout << "[info] Predict Time: " << deltaT56.count() << " s.\n";
    }

    // Compare OpenCL results with reference OpenMP results
    if (param.do_ref == 1) {
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "[info] Computing clMF OpenMP reference results on CPU." << std::endl;
        auto t9 = std::chrono::high_resolution_clock::now();
        ALS_multicore(R, W_ref, H_ref, param);
        auto t10 = std::chrono::high_resolution_clock::now();
        deltaT9_10 = t10 - t9;
        std::cout << "[info] OMP Predict Time: " << deltaT9_10.count() << " s.\n";
        std::cout << "[info] validate the results." << std::endl;
        golden_compare(W_c, W_ref, R.rows, param.k);
        golden_compare(H_c, H_ref, R.cols, param.k);
        calculate_rmse(W_ref, H_ref, param.src_dir, param.k);
    }
    std::cout << "------------------------------------------------------" << std::endl;

    // Some print debugging
//    print_matrix(W_c, R.rows, param.k);
//    print_matrix(H_c, R.cols, param.k);
//
//    print_matrix(W_ref, R.rows, param.k);
//    print_matrix(H_ref, R.cols, param.k);

    auto t8 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT78 = t8 - t7;
    std::cout << "Total Time: " << deltaT78.count() << " Parcial Sums:"
              << deltaT12.count() + deltaT34.count() + deltaT56.count() + deltaTAB.count() + deltaT9_10.count()
              << " s.\n";
    return EXIT_SUCCESS;
}
