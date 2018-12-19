#ifndef PMF_H
#define PMF_H

class parameter {
public:
    int threads = 16;
    int do_ref = 0; // compare opencl results to reference results
    int do_predict = 0;
    int k = 10;
    int maxiter = 5;
    float lambda = 0.05f;
    int nBlocks = 8192;
    int nThreadsPerBlock = 32;
    int platform_id = 0;
    int verbose = 0;
    char device_type[4] = {'g', 'p', 'u', '\0'};
    char opencl_filename[1024] = "../kcode/ALS.cl";
    char src_dir[1024] = "../data/simple";
};

#endif
