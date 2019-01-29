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
                       const unsigned cols) {
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);

    int c = global_id;
    if (c < nnz) {
        for (int t = 0; t < k; t++) {
            unsigned i = test_row[c];
            unsigned j = test_col[c];
            pred_v[c] += W[i * k + t] * H[j * k + t]; //W[i][t] * H[j][t];
//            pred_v[c] += W[t * rows + i] * H[t * cols + j]; //W[i][t] * H[j][t];
        }

        rmse[c] = (pred_v[c] - test_val[c]) * (pred_v[c] - test_val[c]);
    }

//    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
//        barrier(CLK_LOCAL_MEM_FENCE);
//        if (local_id < stride) {
//            rmse[local_id] += rmse[local_id + stride];
//        }
//    }
}

static void choldc1(int n, __global VALUE_TYPE* a, __global VALUE_TYPE* p) {
    int base = get_group_id(0) * n * n;
    int k;
    VALUE_TYPE sum;
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            //sum = a[i][j];
            sum = a[base + i * n + j];
            for (k = i - 1; k >= 0; --k) {
                //sum -= a[i][k] * a[j][k];
                sum -= a[base + i * n + k] * a[base + j * n + k];
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

static void choldcsl(int n, __global VALUE_TYPE* A, __global VALUE_TYPE* tp) {
    VALUE_TYPE sum;
    int base = get_group_id(0) * n * n;
    __global VALUE_TYPE* p;
    int gid = get_group_id(0);
    p = &(tp[gid * n]);
    choldc1(n, A, p);
    for (int i = 0; i < n; ++i) {
        //A[i][i] = 1 / p[i];
        A[base + i * n + i] = 1 / p[i];
        for (int j = i + 1; j < n; ++j) {
            sum = 0;
            for (int k = i; k < j; ++k) {
                //sum -= A[j][k] * A[k][i];
                sum -= A[base + j * n + k] * A[base + k * n + i];
            }
            //A[j][i] = sum / p[j];
            A[base + j * n + i] = sum / p[j];
        }
    }
}

static void inverseMatrix_CholeskyMethod(int n, __global VALUE_TYPE* A, __global VALUE_TYPE* p) {
    int base = get_group_id(0) * n * n;
    int i, j, k;
    choldcsl(n, A, p);
    //vecIndex = (i * 3) + j; to ontain index from vector if needed.
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            //A[i][j] = 0.0;
            A[base + i * n + j] = 0.0;
        }
    }
    for (i = 0; i < n; i++) {
        //A[i][i] *= A[i][i];
        A[base + i * n + i] *= A[base + i * n + i];
        for (k = i + 1; k < n; ++k) {
            //A[i][i] += A[k][i] * A[k][i];
            A[base + i * n + i] += A[base + k * n + i] * A[base + k * n + i];
        }
        for (j = i + 1; j < n; ++j) {
            for (k = j; k < n; ++k) {
                //A[i][j] += A[k][i] * A[k][j];
                A[base + i * n + j] += A[base + k * n + i] * A[base + k * n + j];
            }
        }
    }
    for (i = 0; i < n; ++i) {
        for (j = 0; j < i; ++j) {
            //A[i][j] = A[j][i];
            A[base + i * n + j] = A[base + j * n + i];
        }
    }
}

static void Mt_byM_multiply_k(int i, int j, __global VALUE_TYPE* H, __global VALUE_TYPE* Result, const unsigned ptr,
                              __global const unsigned* idx) {
    int base = get_group_id(0) * j * j;
    int ss = get_local_id(0);
    int gg = get_local_size(0);
    VALUE_TYPE SUM[K_SIZE * K_SIZE] = {0};
    for (int I = ss; I < j; I += gg) {
        for (int J = I; J < j; ++J) {
            for (int K = 0; K < i; ++K) {
                unsigned offset = idx[ptr + K] * j;
                SUM[I * j + J] += H[offset + I] * H[offset + J];
            }
            Result[base + (J * j) + I] = SUM[I * j + J];
            Result[base + (I * j) + J] = SUM[I * j + J];
        }
    }
}

__kernel void updateW_overH_kernel(const int rows,
                                   __global const unsigned* row_ptr,
                                   __global const unsigned* col_idx,
                                   __global const unsigned* colMajored_sparse_idx,
                                   __global const VALUE_TYPE* val,
                                   const VALUE_TYPE lambda,
                                   const uint k,
                                   __global VALUE_TYPE* W,
                                   __global VALUE_TYPE* H,
                                   __global VALUE_TYPE* p,
                                   __global VALUE_TYPE* subVector,
                                   __global VALUE_TYPE* subMatrix,
                                   __global VALUE_TYPE* subMatrix_f) {
    //int i = get_global_id(0);
    //int j = get_global_size(0);
    int s = get_local_id(0);
    int g = get_local_size(0);
    int a = get_group_id(0);
    int v = get_num_groups(0);
    int base = a * k * k;
    int baseV = a * k;
    for (int Rw = a; Rw < rows; Rw += v) {
        __global VALUE_TYPE* Wr = &W[Rw * k];
        unsigned omegaSize = row_ptr[Rw + 1] - row_ptr[Rw];
        //printf("omegasize=%d.\n",omegaSize);
        if (omegaSize > 0) {
            Mt_byM_multiply_k(omegaSize, k, H, subMatrix, row_ptr[Rw], col_idx);
            barrier(CLK_LOCAL_MEM_FENCE);

            for (unsigned c = s; c < k; c += g) {
                subMatrix[base + c * k + c] += lambda;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (s == 0) {
                inverseMatrix_CholeskyMethod(k, subMatrix, p);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (unsigned c = s; c < k; c += g) {
                subVector[baseV + c] = 0;
                for (unsigned idx = row_ptr[Rw]; idx < row_ptr[Rw + 1]; ++idx) {
                    unsigned idx2 = colMajored_sparse_idx[idx];
                    subVector[baseV + c] += val[idx2] * H[(col_idx[idx] * k) + c];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (unsigned c = s; c < k; c += g) {
                Wr[c] = 0.0;
                for (unsigned subVid = 0; subVid < k; ++subVid) {
                    Wr[c] += subVector[baseV + subVid] * subMatrix[base + c * k + subVid];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        } else {
            for (unsigned c = 0; c < k; ++c) {
                Wr[c] = 0.0;
            }
        }
    }
}

__kernel void updateH_overW_kernel(const int cols,
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
    //int i = get_global_id(0);
    //int j = get_global_size(0);
    int s = get_local_id(0);
    int g = get_local_size(0);
    int a = get_group_id(0);
    int v = get_num_groups(0);
    int base = a * k * k;
    int baseV = a * k;
    for (int Rh = a; Rh < cols; Rh += v) {
        __global VALUE_TYPE* Hr = &H[Rh * k];
        unsigned omegaSize = col_ptr[Rh + 1] - col_ptr[Rh];
        if (omegaSize > 0) {
            Mt_byM_multiply_k(omegaSize, k, W, subMatrix, col_ptr[Rh], row_idx);
            barrier(CLK_GLOBAL_MEM_FENCE);

            for (unsigned c = s; c < k; c += g) {
                subMatrix[base + c * k + c] += lambda;
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            if (s == 0) {
                inverseMatrix_CholeskyMethod(k, subMatrix, p);
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            for (unsigned c = s; c < k; c += g) {
                subVector[baseV + c] = 0;
                for (unsigned idx = col_ptr[Rh]; idx < col_ptr[Rh + 1]; ++idx) {
                    subVector[baseV + c] += val[idx] * W[(row_idx[idx] * k) + c];
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            for (unsigned c = s; c < k; c += g) {
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
