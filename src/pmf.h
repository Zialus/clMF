#ifndef PMF_H
#define PMF_H

class parameter {
public:
    int k;
    int maxiter;
    int maxinneriter;
    float lambda;
    int verbose;
    int nBlocks;
    int nThreadsPerBlock;
    int threads;
    int platform_id;

    parameter() {
        k = 10;
        maxiter = 5;
        maxinneriter = 1;
        lambda = 0.05;
        verbose = 0;
        nBlocks = 8192;
        nThreadsPerBlock = 32;
        threads = 4;
        platform_id = 0;
    }
};

#endif
