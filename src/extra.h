#ifndef EXTRA_H
#define EXTRA_H

#include <cmath>
#include <cstdio>
#include <cstdlib>

void choldcsl(int n, float** A);
void choldc1(int n, float** a, float* p);
void inverseMatrix_CholeskyMethod(int n, float** A);
void M_byMt_multiply(int i, int j, float** M, float** Result);
void Mt_byM_multiply(int i, int j, float** M, float** Result);

#endif //EXTRA_H
