#ifndef EXTRA_H
#define EXTRA_H

#include <omp.h>

#include "pmf.h"
#include "pmf_util.h"
#include "tools.h"

void choldcsl(unsigned n, VALUE_TYPE** A);
void choldc1(unsigned n, VALUE_TYPE** a, VALUE_TYPE* p);
void inverseMatrix_CholeskyMethod(unsigned n, VALUE_TYPE** A);
void Mt_byM_multiply(unsigned i, unsigned j, VALUE_TYPE** M, VALUE_TYPE** Result);
void clmf_ref(SparseMatrix& R, MatData& W, MatData& H, TestData& T, parameter& param);

#endif //EXTRA_H
