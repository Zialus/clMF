#ifndef PMF_H
#define PMF_H

class parameter {
public:
    int k;
    int maxiter;
    float lambda;
    int nBlocks;
    int nThreadsPerBlock;
    int platform_id;

    parameter() {
        k = 10;
        maxiter = 5;
        lambda = 0.05;
        nBlocks = 8192;
        nThreadsPerBlock = 32;
        platform_id = 0;
    }
};

#endif
