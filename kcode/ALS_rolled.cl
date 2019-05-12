__kernel void GPU_rmse(__global unsigned const* test_row,
                       __global unsigned const* test_col,
                       __global VALUE_TYPE const* test_val,
                       __global VALUE_TYPE* pred_v,
                       __global VALUE_TYPE* rmse,
                       __global VALUE_TYPE const* W,
                       __global VALUE_TYPE const* H,
                       const unsigned nnz,
                       const unsigned k,
                       const unsigned rows,
                       const unsigned cols,
                       const unsigned isALS) {
    size_t global_id = get_global_id(0);
    size_t global_size = get_global_size(0);
//    size_t local_id = get_local_id(0);
//    size_t group_size = get_local_size(0);

//    size_t c = global_id;
//    if (c < nnz) {
    for (size_t c = global_id; c < nnz; c+=global_size) {
        pred_v[c]=0;
        for (unsigned t = 0; t < k; t++) {
            unsigned i = test_row[c];
            unsigned j = test_col[c];
            if (isALS) {
                pred_v[c] += W[i * k + t] * H[j * k + t]; //W[i][t] * H[j][t];  ALS
            } else {
                pred_v[c] += W[t * rows + i] * H[t * cols + j]; //W[i][t] * H[j][t]; CCD
            }
        }

        rmse[c] = (pred_v[c] - test_val[c]) * (pred_v[c] - test_val[c]);
    }

}

static void choldc1(size_t n, __global VALUE_TYPE* a, __global VALUE_TYPE* p) {
    size_t group_id = get_group_id(0);

    size_t base = group_id * n * n;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            //sum = a[i][j];
            VALUE_TYPE sum = a[base + i * n + j];
            for (int k = (int) i - 1; k >= 0; --k) {
                //sum -= a[i][k] * a[j][k];
                sum -= a[base + i * n + (size_t) k] * a[base + j * n + (size_t) k];
            }
            if (i == j) {
                if (sum <= 0) {
                    printf(" a is not positive definite!\n");
                }
                p[i] = sqrt(sum);
            } else {
                //a[j][i] = sum / p[i];
                a[base + j * n + i] = sum / p[i];
            }
        }
    }
}

static void choldcsl(size_t n, __global VALUE_TYPE* A, __global VALUE_TYPE* tp) {
    size_t group_id = get_group_id(0);
    size_t base = group_id * n * n;

    __global VALUE_TYPE* p;
    p = &(tp[group_id * n]);
    choldc1(n, A, p);

    for (size_t i = 0; i < n; ++i) {
        //A[i][i] = 1 / p[i];
        A[base + i * n + i] = 1 / p[i];
        for (size_t j = i + 1; j < n; ++j) {
            VALUE_TYPE sum = 0;
            for (size_t k = i; k < j; ++k) {
                //sum -= A[j][k] * A[k][i];
                sum -= A[base + j * n + k] * A[base + k * n + i];
            }
            //A[j][i] = sum / p[j];
            A[base + j * n + i] = sum / p[j];
        }
    }
}

static void inverseMatrix_CholeskyMethod(size_t n, __global VALUE_TYPE* A, __global VALUE_TYPE* p) {
    size_t group_id = get_group_id(0);
    size_t base = group_id * n * n;

    choldcsl(n, A, p);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            //A[i][j] = 0.0;
            A[base + i * n + j] = 0.0;
        }
    }
    for (size_t i = 0; i < n; i++) {
        //A[i][i] *= A[i][i];
        A[base + i * n + i] *= A[base + i * n + i];
        for (size_t k = i + 1; k < n; ++k) {
            //A[i][i] += A[k][i] * A[k][i];
            A[base + i * n + i] += A[base + k * n + i] * A[base + k * n + i];
        }
        for (size_t j = i + 1; j < n; ++j) {
            for (size_t k = j; k < n; ++k) {
                //A[i][j] += A[k][i] * A[k][j];
                A[base + i * n + j] += A[base + k * n + i] * A[base + k * n + j];
            }
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < i; ++j) {
            //A[i][j] = A[j][i];
            A[base + i * n + j] = A[base + j * n + i];
        }
    }
}

static void Mt_byM_multiply_k(size_t i, size_t j, __global VALUE_TYPE* H, __global VALUE_TYPE* Result,
                              const unsigned ptr, __global const unsigned* idx) {
    size_t group_id = get_group_id(0);

    size_t base = group_id * j * j;
    size_t ss = get_local_id(0);
    size_t gg = get_local_size(0);
    VALUE_TYPE SUM[K_SIZE * K_SIZE] = {0};
    for (size_t I = ss; I < j; I += gg) {
        for (size_t J = I; J < j; ++J) {
            for (size_t K = 0; K < i; ++K) {
                size_t offset = idx[ptr + K] * j;
                SUM[I * j + J] += H[offset + I] * H[offset + J];
            }
            Result[base + (J * j) + I] = SUM[I * j + J];
            Result[base + (I * j) + J] = SUM[I * j + J];
        }
    }
}

__kernel void updateW_overH_kernel(const uint rows,
                                   __global const unsigned* row_ptr,
                                   __global const unsigned* col_idx,
                                   __global const VALUE_TYPE* val_t,
                                   const VALUE_TYPE lambda,
                                   const uint k,
                                   __global VALUE_TYPE* W,
                                   __global VALUE_TYPE* H,
                                   __global VALUE_TYPE* p,
                                   __global VALUE_TYPE* subVector,
                                   __global VALUE_TYPE* subMatrix,
                                   __global VALUE_TYPE* subMatrix_f) {
    //size_t i = get_global_id(0);
    //size_t j = get_global_size(0);
    size_t s = get_local_id(0);
    size_t g = get_local_size(0);
    size_t a = get_group_id(0);
    size_t v = get_num_groups(0);
    size_t base = a * k * k;
    size_t baseV = a * k;
    for (size_t Rw = a; Rw < rows; Rw += v) {
        __global VALUE_TYPE* Wr = &W[Rw * k];
        unsigned omegaSize = row_ptr[Rw + 1] - row_ptr[Rw];

        if (omegaSize > 0) {
            Mt_byM_multiply_k(omegaSize, k, H, subMatrix, row_ptr[Rw], col_idx);
            barrier(CLK_GLOBAL_MEM_FENCE);

            for (size_t c = s; c < k; c += g) {
                subMatrix[base + c * k + c] += lambda;
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            if (s == 0) {
                inverseMatrix_CholeskyMethod(k, subMatrix, p);
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            /*
            for (unsigned c = s; c < k; c += g) {
                for (unsigned aa = 0; aa < k; aa++) {
                    subMatrix_f[c * k + aa] = subMatrix[base + c * k + aa];
                }
            }
            */
            for (size_t c = s; c < k; c += g) {
                subVector[baseV + c] = 0;
                for (unsigned idx = row_ptr[Rw]; idx < row_ptr[Rw + 1]; ++idx) {
                    subVector[baseV + c] += val_t[idx] * H[(col_idx[idx] * k) + c];
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            for (size_t c = s; c < k; c += g) {
                Wr[c] = 0.0;
                for (unsigned subVid = 0; subVid < k; ++subVid) {
                    Wr[c] += subVector[baseV + subVid] * subMatrix[base + c * k + subVid];
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        } else {
            for (unsigned c = 0; c < k; ++c) {
                Wr[c] = 0.0;
            }
        }
    }
}

__kernel void updateH_overW_kernel(const uint cols,
                                   __global const unsigned* col_ptr,
                                   __global const unsigned* row_idx,
                                   __global const VALUE_TYPE* val,
                                   const VALUE_TYPE lambda,
                                   const uint k,
                                   __global VALUE_TYPE* W,
                                   __global VALUE_TYPE* H,
                                   __global VALUE_TYPE* p,
                                   __global VALUE_TYPE* subVector,
                                   __global VALUE_TYPE* subMatrix) {
    //size_t i = get_global_id(0);
    //size_t j = get_global_size(0);
    size_t s = get_local_id(0);
    size_t g = get_local_size(0);
    size_t a = get_group_id(0);
    size_t v = get_num_groups(0);
    size_t base = a * k * k;
    size_t baseV = a * k;
    for (size_t Rh = a; Rh < cols; Rh += v) {
        __global VALUE_TYPE* Hr = &H[Rh * k];
        unsigned omegaSize = col_ptr[Rh + 1] - col_ptr[Rh];

        if (omegaSize > 0) {
            Mt_byM_multiply_k(omegaSize, k, W, subMatrix, col_ptr[Rh], row_idx);

            barrier(CLK_GLOBAL_MEM_FENCE);

            for (size_t c = s; c < k; c += g) {
                subMatrix[base + c * k + c] += lambda;
            }

            barrier(CLK_GLOBAL_MEM_FENCE);

            if (s == 0) {
                inverseMatrix_CholeskyMethod(k, subMatrix, p);
            }

            barrier(CLK_GLOBAL_MEM_FENCE);

            for (size_t c = s; c < k; c += g) {
                subVector[baseV + c] = 0;
                for (unsigned idx = col_ptr[Rh]; idx < col_ptr[Rh + 1]; ++idx) {
                    subVector[baseV + c] += val[idx] * W[(row_idx[idx] * k) + c];
                }
            }

            barrier(CLK_GLOBAL_MEM_FENCE);

            for (size_t c = s; c < k; c += g) {
                Hr[c] = 0;
                for (unsigned subVid = 0; subVid < k; ++subVid) {
                    Hr[c] += subVector[baseV + subVid] * subMatrix[base + c * k + subVid];
                }
            }

            barrier(CLK_GLOBAL_MEM_FENCE);
        } else {
            for (unsigned c = 0; c < k; ++c) {
                Hr[c] = 0.0;
            }
        }
    }
}
