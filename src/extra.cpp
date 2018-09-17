#include "extra.h"

void choldc1(int n, float** a, float* p) {
    unsigned i, j;
    int k;
    float sum;
    for (i = 0; i < n; ++i) {
        for (j = i; j < n; ++j) {
            sum = a[i][j];
            for (k = i - 1; k >= 0; --k) {
                sum -= a[i][k] * a[j][k];
            }
            if (i == j) {
                if (sum <= 0) {
                    printf(" a is not positive definite!\n");
                }
                p[i] = sqrtf(sum);//float square root
            } else {
                a[j][i] = sum / p[i];
            }
        }
    }
}

void choldcsl(int n, float** A) {
    unsigned i, j, k;
    double sum;
    float* p;
    p = (float*) malloc(n * sizeof(float));
    choldc1(n, A, p);
    for (i = 0; i < n; ++i) {
        A[i][i] = 1 / p[i];
        for (j = i + 1; j < n; ++j) {
            sum = 0;
            for (k = i; k < j; ++k) {
                sum -= A[j][k] * A[k][i];
            }
            A[j][i] = sum / p[j];
        }
    }
    free(p);
}

void inverseMatrix_CholeskyMethod(int n, float** A) {
    unsigned i, j, k;
    choldcsl(n, A);
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            A[i][j] = 0.0;
        }
    }
    for (i = 0; i < n; i++) {
        A[i][i] *= A[i][i];
        for (k = i + 1; k < n; ++k) {
            A[i][i] += A[k][i] * A[k][i];
        }
        for (j = i + 1; j < n; ++j) {
            for (k = j; k < n; ++k) {
                A[i][j] += A[k][i] * A[k][j];
            }
        }
    }
    for (i = 0; i < n; ++i) {
        for (j = 0; j < i; ++j) {
            A[i][j] = A[j][i];
        }
    }
}

//Multiply matrix M by M transpose
void M_byMt_multiply(int i, int j, float** M, float** Result) {
    float SUM;
    for (unsigned I = 0; I < i; ++I) {
        for (unsigned J = 0; J < i; ++J) {
            SUM = 0.0;
            for (unsigned K = 0; K < j; ++K) {
                SUM += M[I][K] * M[J][K];
            }
            Result[I][J] = SUM;
        }
    }
}

//Multiply matrix M transpose by M
void Mt_byM_multiply(int i, int j, float** M, float** Result) {
    float SUM;
    for (unsigned I = 0; I < j; ++I) {
        for (unsigned J = I; J < j; ++J) {
            SUM = 0.0f;
            for (unsigned K = 0; K < i; ++K) {
                //printf("%.3f %.3f\n", M[K][I], M[K][J]);
                SUM += M[K][I] * M[K][J];
            }
            Result[J][I] = SUM;
            Result[I][J] = SUM;
        }
    }
}
