#ifndef EXTRA_H
#define EXTRA_H

#include <omp.h>

#include "pmf.h"
#include "pmf_util.h"

void choldcsl(int n, float** A);
void choldc1(int n, float** a, float* p);
void inverseMatrix_CholeskyMethod(int n, float** A);
void M_byMt_multiply(int i, int j, float** M, float** Result);
void Mt_byM_multiply(int i, int j, float** M, float** Result);
void clmf_ref(smat_t& R, mat_t& W, mat_t& H, parameter& param);

#endif //EXTRA_H
