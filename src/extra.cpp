#include "extra.h"

void choldc1(int n, float** a, float* p) {
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            float sum = a[i][j];
            for (int k = i - 1; k >= 0; --k) {
                sum -= a[i][k] * a[j][k];
            }
            if (i == j) {
                if (sum <= 0) {
                    printf(" a is not positive definite!\n");
                }
                p[i] = sqrtf(sum);
            } else {
                a[j][i] = sum / p[i];
            }
        }
    }
}

void choldcsl(int n, float** A) {
    float* p = (float*) malloc(n * sizeof(float));
    choldc1(n, A, p);
    for (int i = 0; i < n; ++i) {
        A[i][i] = 1 / p[i];
        for (int j = i + 1; j < n; ++j) {
            float sum = 0;
            for (int k = i; k < j; ++k) {
                sum -= A[j][k] * A[k][i];
            }
            A[j][i] = sum / p[j];
        }
    }
    free(p);
}

void inverseMatrix_CholeskyMethod(int n, float** A) {
    choldcsl(n, A);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            A[i][j] = 0.0;
        }
    }
    for (int i = 0; i < n; i++) {
        A[i][i] *= A[i][i];
        for (int k = i + 1; k < n; ++k) {
            A[i][i] += A[k][i] * A[k][i];
        }
        for (int j = i + 1; j < n; ++j) {
            for (int k = j; k < n; ++k) {
                A[i][j] += A[k][i] * A[k][j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i][j] = A[j][i];
        }
    }
}

//Multiply matrix M by M transpose
void M_byMt_multiply(int i, int j, float** M, float** Result) {
    float SUM;
    for (int I = 0; I < i; ++I) {
        for (int J = 0; J < i; ++J) {
            SUM = 0.0;
            for (int K = 0; K < j; ++K) {
                SUM += M[I][K] * M[J][K];
            }
            Result[I][J] = SUM;
        }
    }
}

//Multiply matrix M transpose by M
void Mt_byM_multiply(int i, int j, float** M, float** Result) {
    float SUM;
    for (int I = 0; I < j; ++I) {
        for (int J = I; J < j; ++J) {
            SUM = 0.0f;
            for (int K = 0; K < i; ++K) {
                //printf("%.3f %.3f\n", M[K][I], M[K][J]);
                SUM += M[K][I] * M[K][J];
            }
            Result[J][I] = SUM;
            Result[I][J] = SUM;
        }
    }
}

#define kind dynamic,500

void ALS_multicore(smat_t& R, mat_t& W, mat_t& H, parameter& param) {
    int maxIter = param.maxiter;
    float lambda = param.lambda;
    int k = param.k;
    int num_threads_old = omp_get_num_threads();

    omp_set_num_threads(param.threads);

    for (int iter = 0; iter < maxIter; ++iter) {

        //optimize W over H
#pragma omp parallel for schedule(kind)
        for (long Rw = 0; Rw < R.rows; ++Rw) {
            float* Wr = &W[Rw][0];
            int omegaSize = R.row_ptr[Rw + 1] - R.row_ptr[Rw];
            float** subMatrix;

            if (omegaSize > 0) {
                float* subVector = (float*) malloc(k * sizeof(float));
                subMatrix = (float**) malloc(k * sizeof(float*));
                for (int i = 0; i < k; ++i) {
                    subMatrix[i] = (float*) malloc(k * sizeof(float));
                }

                //a trick to avoid malloc
                float** H_Omega = (float**) malloc(omegaSize * sizeof(float*));
                for (unsigned idx = R.row_ptr[Rw], i = 0; idx < R.row_ptr[Rw + 1]; ++idx, ++i) {
                    H_Omega[i] = &H[R.col_idx[idx]][0];
                }

                Mt_byM_multiply(omegaSize, k, H_Omega, subMatrix);

                //add lambda to diag of sub-matrix
                for (int c = 0; c < k; c++) {
                    subMatrix[c][c] = subMatrix[c][c] + lambda;
                }

                //invert sub-matrix
                inverseMatrix_CholeskyMethod(k, subMatrix);


                //sparse multiplication
                for (int c = 0; c < k; ++c) {
                    subVector[c] = 0;
                    for (unsigned idx = R.row_ptr[Rw]; idx < R.row_ptr[Rw + 1]; ++idx) {
                        unsigned idx2 = R.colMajored_sparse_idx[idx];
                        subVector[c] += R.val[idx2] * H[R.col_idx[idx]][c];
                    }
                }

                //multiply subVector by subMatrix
                for (int c = 0; c < k; ++c) {
                    Wr[c] = 0;
                    for (int subVid = 0; subVid < k; ++subVid) {
                        Wr[c] += subVector[subVid] * subMatrix[c][subVid];
                    }
                }


                for (int i = 0; i < k; ++i) {
                    free(subMatrix[i]);
                }
                free(subMatrix);
                free(subVector);
                free(H_Omega);
            } else {
                for (int c = 0; c < k; ++c) {
                    Wr[c] = 0.0f;
                    //printf("%.3f ", Wr[c]);
                }
                //printf("\n");
            }
        }

        //optimize H over W
#pragma omp parallel for schedule(kind)
        for (long Rh = 0; Rh < R.cols; ++Rh) {
            float* Hr = &H[Rh][0];
            unsigned omegaSize = R.col_ptr[Rh + 1] - R.col_ptr[Rh];
            float** subMatrix;

            if (omegaSize > 0) {
                float* subVector = (float*) malloc(k * sizeof(float));
                subMatrix = (float**) malloc(k * sizeof(float*));
                for (int i = 0; i < k; ++i) {
                    subMatrix[i] = (float*) malloc(k * sizeof(float));
                }

                //a trick to avoid malloc
                float** W_Omega = (float**) malloc(omegaSize * sizeof(float*));
                for (unsigned idx = R.col_ptr[Rh], i = 0; idx < R.col_ptr[Rh + 1]; ++idx, ++i) {
                    W_Omega[i] = &W[R.row_idx[idx]][0];
                }

                Mt_byM_multiply(omegaSize, k, W_Omega, subMatrix);

                //add lambda to diag of sub-matrix
                for (int c = 0; c < k; c++) {
                    subMatrix[c][c] = subMatrix[c][c] + lambda;
                }

                //invert sub-matrix
                inverseMatrix_CholeskyMethod(k, subMatrix);


                //sparse multiplication
                for (int c = 0; c < k; ++c) {
                    subVector[c] = 0;
                    for (unsigned idx = R.col_ptr[Rh]; idx < R.col_ptr[Rh + 1]; ++idx) {
                        subVector[c] += R.val[idx] * W[R.row_idx[idx]][c];
                    }
                }

                //multiply subVector by subMatrix
                for (int c = 0; c < k; ++c) {
                    Hr[c] = 0;
                    for (int subVid = 0; subVid < k; ++subVid) {
                        Hr[c] += subVector[subVid] * subMatrix[c][subVid];
                    }
                }


                for (int i = 0; i < k; ++i) {
                    free(subMatrix[i]);
                }
                free(subMatrix);
                free(subVector);
                free(W_Omega);
            } else {
                for (int c = 0; c < k; ++c) {
                    Hr[c] = 0.0f;
                }
            }
        }

    }
    omp_set_num_threads(num_threads_old);
}
