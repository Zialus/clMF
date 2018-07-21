#ifndef PMF_H
#define PMF_H

class parameter {
public:
    int solver_type;
    int k;
    int maxiter;
    int maxinneriter;
    float lambda;
    float rho;
    int lrate_method;
    int num_blocks;
    int do_predict;
    int verbose;
    int do_nmf;  // non-negative matrix factorization
    int nBlocks;
    int nThreadsPerBlock;

    parameter() {
        k = 10;
        maxiter = 5;
        lambda = 0.1f;
        do_predict = 0;
        verbose = 0;
        do_nmf = 0;
        nBlocks = 8192;
        nThreadsPerBlock = 32;
    }
};

#endif
