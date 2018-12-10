#ifndef PMF_H
#define PMF_H

class parameter {
public:
    int k;
    int maxiter;
    float lambda;
    int nBlocks;
    int nThreadsPerBlock;
    int device_id;
    int platform_id;
    int verbose;

    parameter() {
        k = 10;
        maxiter = 5;
        lambda = 0.05f;
        nBlocks = 8192;
        nThreadsPerBlock = 32;
        device_id = 0;
        platform_id = 0;
        verbose = 0;
    }
};

#endif
