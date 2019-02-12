#include "clmf_ref.h"

void choldc1(int n, VALUE_TYPE** a, VALUE_TYPE* p) {
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            VALUE_TYPE sum = a[i][j];
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

void choldcsl(int n, VALUE_TYPE** A) {
    VALUE_TYPE* p = (VALUE_TYPE*) malloc(n * sizeof(VALUE_TYPE));
    choldc1(n, A, p);

    for (int i = 0; i < n; ++i) {
        A[i][i] = 1 / p[i];
        for (int j = i + 1; j < n; ++j) {
            VALUE_TYPE sum = 0;
            for (int k = i; k < j; ++k) {
                sum -= A[j][k] * A[k][i];
            }
            A[j][i] = sum / p[j];
        }
    }
    free(p);
}

void inverseMatrix_CholeskyMethod(int n, VALUE_TYPE** A) {
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

void Mt_byM_multiply(int i, int j, VALUE_TYPE** M, VALUE_TYPE** Result) {
    VALUE_TYPE SUM;
    for (int I = 0; I < j; ++I) {
        for (int J = I; J < j; ++J) {
            SUM = 0.0;
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

void clmf_ref(smat_t& R, mat_t& W, mat_t& H, testset_t& T,parameter& param) {
    int k = (int) param.k;

    int num_threads_old = omp_get_num_threads();
    omp_set_num_threads(param.threads);

    double update_time_acc = 0;

    for (int iter = 0; iter < param.maxiter; ++iter) {

        double start = omp_get_wtime();

        //optimize W over H
#pragma omp parallel for schedule(kind)
        for (long Rw = 0; Rw < R.rows; ++Rw) {
            VALUE_TYPE* Wr = &W[Rw][0];
            int omegaSize = (int) R.row_ptr[Rw + 1] - R.row_ptr[Rw];
            VALUE_TYPE** subMatrix;

            if (omegaSize > 0) {
                VALUE_TYPE* subVector = (VALUE_TYPE*) malloc(k * sizeof(VALUE_TYPE));
                subMatrix = (VALUE_TYPE**) malloc(k * sizeof(VALUE_TYPE*));
                for (int i = 0; i < k; ++i) {
                    subMatrix[i] = (VALUE_TYPE*) malloc(k * sizeof(VALUE_TYPE));
                }

                //a trick to avoid malloc
                VALUE_TYPE** H_Omega = (VALUE_TYPE**) malloc(omegaSize * sizeof(VALUE_TYPE*));
                for (unsigned idx = R.row_ptr[Rw], i = 0; idx < R.row_ptr[Rw + 1]; ++idx, ++i) {
                    H_Omega[i] = &H[R.col_idx[idx]][0];
                }

                Mt_byM_multiply(omegaSize, k, H_Omega, subMatrix);

                //add lambda to diag of sub-matrix
                for (int c = 0; c < k; c++) {
                    subMatrix[c][c] = subMatrix[c][c] + param.lambda;
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
                    Wr[c] = 0.0;
                    //printf("%.3f ", Wr[c]);
                }
                //printf("\n");
            }
        }

        //optimize H over W
#pragma omp parallel for schedule(kind)
        for (long Rh = 0; Rh < R.cols; ++Rh) {
            VALUE_TYPE* Hr = &H[Rh][0];
            int omegaSize = (int) R.col_ptr[Rh + 1] - R.col_ptr[Rh];
            VALUE_TYPE** subMatrix;

            if (omegaSize > 0) {
                VALUE_TYPE* subVector = (VALUE_TYPE*) malloc(k * sizeof(VALUE_TYPE));
                subMatrix = (VALUE_TYPE**) malloc(k * sizeof(VALUE_TYPE*));
                for (int i = 0; i < k; ++i) {
                    subMatrix[i] = (VALUE_TYPE*) malloc(k * sizeof(VALUE_TYPE));
                }

                //a trick to avoid malloc
                VALUE_TYPE** W_Omega = (VALUE_TYPE**) malloc(omegaSize * sizeof(VALUE_TYPE*));
                for (unsigned idx = R.col_ptr[Rh], i = 0; idx < R.col_ptr[Rh + 1]; ++idx, ++i) {
                    W_Omega[i] = &W[R.row_idx[idx]][0];
                }

                Mt_byM_multiply(omegaSize, k, W_Omega, subMatrix);

                //add lambda to diag of sub-matrix
                for (int c = 0; c < k; c++) {
                    subMatrix[c][c] = subMatrix[c][c] + param.lambda;
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
                    Hr[c] = 0.0;
                }
            }
        }
        double end = omp_get_wtime();
        double update_time = end - start;
        update_time_acc+=update_time;

        start = omp_get_wtime();
        double rmse = calculate_rmse_directly(W, H, T, param.k, true);
        end = omp_get_wtime();
        double rmse_timer = end - start;

        printf("[-INFO-] iteration num %d \tupdate_time %.4lf|%.4lfs \tRMSE=%lf time:%fs\n", iter+1, update_time, update_time_acc, rmse, rmse_timer);

    }
    omp_set_num_threads(num_threads_old);
}
