#ifndef PMF_H
#define PMF_H

class parameter {
public:
    int threads = 16;
    int nBlocks = 8192;
    int nThreadsPerBlock = 32;

    int do_ref = 0; // compare opencl results to reference results
    int do_predict = 0;

    int maxiter = 5;
    unsigned k = 10;
    VALUE_TYPE lambda = (VALUE_TYPE) 0.05;

    unsigned platform_id = 0;
    int verbose = 0;

    int version = 1;

    char device_type[4] = {'g', 'p', 'u', '\0'};
    char src_dir[1024] = "../data/simple";
    char kcode_path[1024] = "../kcode";
};

#endif
