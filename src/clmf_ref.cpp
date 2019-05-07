#include "clmf_ref.h"

void choldc1(unsigned n, VALUE_TYPE** a, VALUE_TYPE* p) {
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = i; j < n; ++j) {
            VALUE_TYPE sum = a[i][j];
            for (int k = (int) i - 1; k >= 0; --k) {
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

void choldcsl(unsigned n, VALUE_TYPE** A) {
    VALUE_TYPE* p = (VALUE_TYPE*) malloc(n * sizeof(VALUE_TYPE));
    choldc1(n, A, p);

    for (unsigned i = 0; i < n; ++i) {
        A[i][i] = 1 / p[i];
        for (unsigned j = i + 1; j < n; ++j) {
            VALUE_TYPE sum = 0;
            for (unsigned k = i; k < j; ++k) {
                sum -= A[j][k] * A[k][i];
            }
            A[j][i] = sum / p[j];
        }
    }
    free(p);
}

void inverseMatrix_CholeskyMethod(unsigned n, VALUE_TYPE** A) {
    choldcsl(n, A);
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = i + 1; j < n; ++j) {
            A[i][j] = 0.0;
        }
    }
    for (unsigned i = 0; i < n; i++) {
        A[i][i] *= A[i][i];
        for (unsigned k = i + 1; k < n; ++k) {
            A[i][i] += A[k][i] * A[k][i];
        }
        for (unsigned j = i + 1; j < n; ++j) {
            for (unsigned k = j; k < n; ++k) {
                A[i][j] += A[k][i] * A[k][j];
            }
        }
    }
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = 0; j < i; ++j) {
            A[i][j] = A[j][i];
        }
    }
}

void Mt_byM_multiply(unsigned i, unsigned j, VALUE_TYPE** M, VALUE_TYPE** Result) {
    VALUE_TYPE SUM;
    for (unsigned I = 0; I < j; ++I) {
        for (unsigned J = I; J < j; ++J) {
            SUM = 0.0;
            for (unsigned K = 0; K < i; ++K) {
                //printf("%.3f %.3f\n", M[K][I], M[K][J]);
                SUM += M[K][I] * M[K][J];
            }
            Result[J][I] = SUM;
            Result[I][J] = SUM;
        }
    }
}

#define kind dynamic,500

void clmf_ref(SparseMatrix& R, MatData& W, MatData& H, TestData& T,parameter& param) {
    unsigned k = param.k;

    int num_threads_old = omp_get_num_threads();
    omp_set_num_threads(param.threads);

    double update_time_acc = 0;

    for (int iter = 0; iter < param.maxiter; ++iter) {

        double start = omp_get_wtime();

        //optimize W over H
#pragma omp parallel for schedule(kind)
        for (long Rw = 0; Rw < R.rows; ++Rw) {
            VALUE_TYPE* Wr = &W[Rw][0];
            unsigned omegaSize = R.get_csr_row_ptr()[Rw + 1] - R.get_csr_row_ptr()[Rw];
            VALUE_TYPE** subMatrix;

            if (omegaSize > 0) {
                VALUE_TYPE* subVector = (VALUE_TYPE*) malloc(k * sizeof(VALUE_TYPE));
                subMatrix = (VALUE_TYPE**) malloc(k * sizeof(VALUE_TYPE*));
                for (unsigned i = 0; i < k; ++i) {
                    subMatrix[i] = (VALUE_TYPE*) malloc(k * sizeof(VALUE_TYPE));
                }

                //a trick to avoid malloc
                VALUE_TYPE** H_Omega = (VALUE_TYPE**) malloc(omegaSize * sizeof(VALUE_TYPE*));
                for (unsigned idx = R.get_csr_row_ptr()[Rw], i = 0; i < omegaSize; ++idx, ++i) {
                    H_Omega[i] = &H[R.get_csr_col_indx()[idx]][0];
                }

                Mt_byM_multiply(omegaSize, k, H_Omega, subMatrix);

                //add lambda to diag of sub-matrix
                for (unsigned c = 0; c < k; c++) {
                    subMatrix[c][c] = subMatrix[c][c] + param.lambda;
                }

                //invert sub-matrix
                inverseMatrix_CholeskyMethod(k, subMatrix);


                //sparse multiplication
                for (unsigned c = 0; c < k; ++c) {
                    subVector[c] = 0;
                    for (unsigned idx = R.get_csr_row_ptr()[Rw]; idx < R.get_csr_row_ptr()[Rw + 1]; ++idx) {
                        subVector[c] += R.get_csr_val()[idx] * H[R.get_csr_col_indx()[idx]][c];
                    }
                }

                //multiply subVector by subMatrix
                for (unsigned c = 0; c < k; ++c) {
                    Wr[c] = 0;
                    for (unsigned subVid = 0; subVid < k; ++subVid) {
                        Wr[c] += subVector[subVid] * subMatrix[c][subVid];
                    }
                }


                for (unsigned i = 0; i < k; ++i) {
                    free(subMatrix[i]);
                }
                free(subMatrix);
                free(subVector);
                free(H_Omega);
            } else {
                for (unsigned c = 0; c < k; ++c) {
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
            unsigned omegaSize = R.get_csc_col_ptr()[Rh + 1] - R.get_csc_col_ptr()[Rh];
            VALUE_TYPE** subMatrix;

            if (omegaSize > 0) {
                VALUE_TYPE* subVector = (VALUE_TYPE*) malloc(k * sizeof(VALUE_TYPE));
                subMatrix = (VALUE_TYPE**) malloc(k * sizeof(VALUE_TYPE*));
                for (unsigned i = 0; i < k; ++i) {
                    subMatrix[i] = (VALUE_TYPE*) malloc(k * sizeof(VALUE_TYPE));
                }

                //a trick to avoid malloc
                VALUE_TYPE** W_Omega = (VALUE_TYPE**) malloc(omegaSize * sizeof(VALUE_TYPE*));
                for (unsigned idx = R.get_csc_col_ptr()[Rh], i = 0; i < omegaSize ; ++idx, ++i) {
                    W_Omega[i] = &W[R.get_csc_row_indx()[idx]][0];
                }

                Mt_byM_multiply(omegaSize, k, W_Omega, subMatrix);

                //add lambda to diag of sub-matrix
                for (unsigned c = 0; c < k; c++) {
                    subMatrix[c][c] = subMatrix[c][c] + param.lambda;
                }

                //invert sub-matrix
                inverseMatrix_CholeskyMethod(k, subMatrix);


                //sparse multiplication
                for (unsigned c = 0; c < k; ++c) {
                    subVector[c] = 0;
                    for (unsigned idx = R.get_csc_col_ptr()[Rh]; idx < R.get_csc_col_ptr()[Rh + 1]; ++idx) {
                        subVector[c] += R.get_csc_val()[idx] * W[R.get_csc_row_indx()[idx]][c];
                    }
                }

                //multiply subVector by subMatrix
                for (unsigned c = 0; c < k; ++c) {
                    Hr[c] = 0;
                    for (unsigned subVid = 0; subVid < k; ++subVid) {
                        Hr[c] += subVector[subVid] * subMatrix[c][subVid];
                    }
                }


                for (unsigned i = 0; i < k; ++i) {
                    free(subMatrix[i]);
                }
                free(subMatrix);
                free(subVector);
                free(W_Omega);
            } else {
                for (unsigned c = 0; c < k; ++c) {
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
