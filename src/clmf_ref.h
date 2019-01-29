#ifndef EXTRA_H
#define EXTRA_H

#include <omp.h>

#include "pmf.h"
#include "pmf_util.h"
#include "tools.h"

void choldcsl(int n, VALUE_TYPE** A);
void choldc1(int n, VALUE_TYPE** a, VALUE_TYPE* p);
void inverseMatrix_CholeskyMethod(int n, VALUE_TYPE** A);
void M_byMt_multiply(int i, int j, VALUE_TYPE** M, VALUE_TYPE** Result);
void Mt_byM_multiply(int i, int j, VALUE_TYPE** M, VALUE_TYPE** Result);
void clmf_ref(smat_t& R, mat_t& W, mat_t& H, testset_t& T, parameter& param);

#endif //EXTRA_H
