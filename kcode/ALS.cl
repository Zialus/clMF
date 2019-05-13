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
    for (size_t c = global_id; c < nnz; c += global_size) {
        pred_v[c] = 0;
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

__kernel void Mt_byM_multiply_k(uint i, uint j, __global VALUE_TYPE* H, __global VALUE_TYPE* Result,
                                const unsigned ptr, __global const unsigned* idx) {
    size_t group_id = get_group_id(0);
    size_t base = group_id * j * j;
    size_t local_id = get_local_id(0);
    size_t local_size = get_local_size(0);

    //__local VALUE_TYPE SUM[100];
    VALUE_TYPE SUM0 = 0, SUM1 = 0, SUM2 = 0, SUM3 = 0, SUM4 = 0, SUM5 = 0, SUM6 = 0, SUM7 = 0, SUM8 = 0, SUM9 = 0,
            SUM11 = 0, SUM12 = 0, SUM13 = 0, SUM14 = 0, SUM15 = 0, SUM16 = 0, SUM17 = 0, SUM18 = 0, SUM19 = 0,
            SUM22 = 0, SUM23 = 0, SUM24 = 0, SUM25 = 0, SUM26 = 0, SUM27 = 0, SUM28 = 0, SUM29 = 0,
            SUM33 = 0, SUM34 = 0, SUM35 = 0, SUM36 = 0, SUM37 = 0, SUM38 = 0, SUM39 = 0,
            SUM44 = 0, SUM45 = 0, SUM46 = 0, SUM47 = 0, SUM48 = 0, SUM49 = 0,
            SUM55 = 0, SUM56 = 0, SUM57 = 0, SUM58 = 0, SUM59 = 0,
            SUM66 = 0, SUM67 = 0, SUM68 = 0, SUM69 = 0,
            SUM77 = 0, SUM78 = 0, SUM79 = 0,
            SUM88 = 0, SUM89 = 0,
            SUM99 = 0;
    __local VALUE_TYPE a[300];
    __local unsigned offset[30];
    unsigned f = 300 / j;
    unsigned nh = (i / f) + 1;
    unsigned p = nh;
    if (i > f) {
        for (; p > 1; p--) {
            for (size_t K = local_id; K < f; K += local_size) {
                offset[K] = idx[ptr + K + (nh - p) * f] * j;
                for (unsigned I = 0; I < j; ++I) {
                    a[K * j + I] = H[offset[K] + I];
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            for (unsigned S = 0; S < f; S++) {
                SUM0 += a[S * j] * a[S * j];
                SUM1 += a[S * j] * a[S * j + 1];
                SUM2 += a[S * j] * a[S * j + 2];
                SUM3 += a[S * j] * a[S * j + 3];
                SUM4 += a[S * j] * a[S * j + 4];
                SUM5 += a[S * j] * a[S * j + 5];
                SUM6 += a[S * j] * a[S * j + 6];
                SUM7 += a[S * j] * a[S * j + 7];
                SUM8 += a[S * j] * a[S * j + 8];
                SUM9 += a[S * j] * a[S * j + 9];

                SUM11 += a[S * j + 1] * a[S * j + 1];
                SUM12 += a[S * j + 1] * a[S * j + 2];
                SUM13 += a[S * j + 1] * a[S * j + 3];
                SUM14 += a[S * j + 1] * a[S * j + 4];
                SUM15 += a[S * j + 1] * a[S * j + 5];
                SUM16 += a[S * j + 1] * a[S * j + 6];
                SUM17 += a[S * j + 1] * a[S * j + 7];
                SUM18 += a[S * j + 1] * a[S * j + 8];
                SUM19 += a[S * j + 1] * a[S * j + 9];

                SUM22 += a[S * j + 2] * a[S * j + 2];
                SUM23 += a[S * j + 2] * a[S * j + 3];
                SUM24 += a[S * j + 2] * a[S * j + 4];
                SUM25 += a[S * j + 2] * a[S * j + 5];
                SUM26 += a[S * j + 2] * a[S * j + 6];
                SUM27 += a[S * j + 2] * a[S * j + 7];
                SUM28 += a[S * j + 2] * a[S * j + 8];
                SUM29 += a[S * j + 2] * a[S * j + 9];

                SUM33 += a[S * j + 3] * a[S * j + 3];
                SUM34 += a[S * j + 3] * a[S * j + 4];
                SUM35 += a[S * j + 3] * a[S * j + 5];
                SUM36 += a[S * j + 3] * a[S * j + 6];
                SUM37 += a[S * j + 3] * a[S * j + 7];
                SUM38 += a[S * j + 3] * a[S * j + 8];
                SUM39 += a[S * j + 3] * a[S * j + 9];

                SUM44 += a[S * j + 4] * a[S * j + 4];
                SUM45 += a[S * j + 4] * a[S * j + 5];
                SUM46 += a[S * j + 4] * a[S * j + 6];
                SUM47 += a[S * j + 4] * a[S * j + 7];
                SUM48 += a[S * j + 4] * a[S * j + 8];
                SUM49 += a[S * j + 4] * a[S * j + 9];

                SUM55 += a[S * j + 5] * a[S * j + 5];
                SUM56 += a[S * j + 5] * a[S * j + 6];
                SUM57 += a[S * j + 5] * a[S * j + 7];
                SUM58 += a[S * j + 5] * a[S * j + 8];
                SUM59 += a[S * j + 5] * a[S * j + 9];

                SUM66 += a[S * j + 6] * a[S * j + 6];
                SUM67 += a[S * j + 6] * a[S * j + 7];
                SUM68 += a[S * j + 6] * a[S * j + 8];
                SUM69 += a[S * j + 6] * a[S * j + 9];

                SUM77 += a[S * j + 7] * a[S * j + 7];
                SUM78 += a[S * j + 7] * a[S * j + 8];
                SUM79 += a[S * j + 7] * a[S * j + 9];

                SUM88 += a[S * j + 8] * a[S * j + 8];
                SUM89 += a[S * j + 8] * a[S * j + 9];

                SUM99 += a[S * j + 9] * a[S * j + 9];
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }

        for (size_t K = local_id; K < i - (nh - 1) * f; K += local_size) {
            offset[K] = idx[ptr + K + (nh - 1) * f] * j;
            for (unsigned I = 0; I < j; ++I) {
                a[K * j + I] = H[offset[K] + I];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        for (unsigned S = 0; S < i - (nh - 1) * f; S++) {
            SUM0 += a[S * j] * a[S * j];
            SUM1 += a[S * j] * a[S * j + 1];
            SUM2 += a[S * j] * a[S * j + 2];
            SUM3 += a[S * j] * a[S * j + 3];
            SUM4 += a[S * j] * a[S * j + 4];
            SUM5 += a[S * j] * a[S * j + 5];
            SUM6 += a[S * j] * a[S * j + 6];
            SUM7 += a[S * j] * a[S * j + 7];
            SUM8 += a[S * j] * a[S * j + 8];
            SUM9 += a[S * j] * a[S * j + 9];

            SUM11 += a[S * j + 1] * a[S * j + 1];
            SUM12 += a[S * j + 1] * a[S * j + 2];
            SUM13 += a[S * j + 1] * a[S * j + 3];
            SUM14 += a[S * j + 1] * a[S * j + 4];
            SUM15 += a[S * j + 1] * a[S * j + 5];
            SUM16 += a[S * j + 1] * a[S * j + 6];
            SUM17 += a[S * j + 1] * a[S * j + 7];
            SUM18 += a[S * j + 1] * a[S * j + 8];
            SUM19 += a[S * j + 1] * a[S * j + 9];

            SUM22 += a[S * j + 2] * a[S * j + 2];
            SUM23 += a[S * j + 2] * a[S * j + 3];
            SUM24 += a[S * j + 2] * a[S * j + 4];
            SUM25 += a[S * j + 2] * a[S * j + 5];
            SUM26 += a[S * j + 2] * a[S * j + 6];
            SUM27 += a[S * j + 2] * a[S * j + 7];
            SUM28 += a[S * j + 2] * a[S * j + 8];
            SUM29 += a[S * j + 2] * a[S * j + 9];

            SUM33 += a[S * j + 3] * a[S * j + 3];
            SUM34 += a[S * j + 3] * a[S * j + 4];
            SUM35 += a[S * j + 3] * a[S * j + 5];
            SUM36 += a[S * j + 3] * a[S * j + 6];
            SUM37 += a[S * j + 3] * a[S * j + 7];
            SUM38 += a[S * j + 3] * a[S * j + 8];
            SUM39 += a[S * j + 3] * a[S * j + 9];

            SUM44 += a[S * j + 4] * a[S * j + 4];
            SUM45 += a[S * j + 4] * a[S * j + 5];
            SUM46 += a[S * j + 4] * a[S * j + 6];
            SUM47 += a[S * j + 4] * a[S * j + 7];
            SUM48 += a[S * j + 4] * a[S * j + 8];
            SUM49 += a[S * j + 4] * a[S * j + 9];

            SUM55 += a[S * j + 5] * a[S * j + 5];
            SUM56 += a[S * j + 5] * a[S * j + 6];
            SUM57 += a[S * j + 5] * a[S * j + 7];
            SUM58 += a[S * j + 5] * a[S * j + 8];
            SUM59 += a[S * j + 5] * a[S * j + 9];

            SUM66 += a[S * j + 6] * a[S * j + 6];
            SUM67 += a[S * j + 6] * a[S * j + 7];
            SUM68 += a[S * j + 6] * a[S * j + 8];
            SUM69 += a[S * j + 6] * a[S * j + 9];

            SUM77 += a[S * j + 7] * a[S * j + 7];
            SUM78 += a[S * j + 7] * a[S * j + 8];
            SUM79 += a[S * j + 7] * a[S * j + 9];

            SUM88 += a[S * j + 8] * a[S * j + 8];
            SUM89 += a[S * j + 8] * a[S * j + 9];

            SUM99 += a[S * j + 9] * a[S * j + 9];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        Result[base + 0] = SUM0;
        Result[base + 1] = SUM1;
        Result[base + 2] = SUM2;
        Result[base + 3] = SUM3;
        Result[base + 4] = SUM4;
        Result[base + 5] = SUM5;
        Result[base + 6] = SUM6;
        Result[base + 7] = SUM7;
        Result[base + 8] = SUM8;
        Result[base + 9] = SUM9;
        Result[base + 10] = Result[base + 1];
        Result[base + 11] = SUM11;
        Result[base + 12] = SUM12;
        Result[base + 13] = SUM13;
        Result[base + 14] = SUM14;
        Result[base + 15] = SUM15;
        Result[base + 16] = SUM16;
        Result[base + 17] = SUM17;
        Result[base + 18] = SUM18;
        Result[base + 19] = SUM19;
        Result[base + 20] = Result[base + 2];
        Result[base + 21] = Result[base + 12];
        Result[base + 22] = SUM22;
        Result[base + 23] = SUM23;
        Result[base + 24] = SUM24;
        Result[base + 25] = SUM25;
        Result[base + 26] = SUM26;
        Result[base + 27] = SUM27;
        Result[base + 28] = SUM28;
        Result[base + 29] = SUM29;
        Result[base + 30] = Result[base + 3];
        Result[base + 31] = Result[base + 13];
        Result[base + 32] = Result[base + 23];
        Result[base + 33] = SUM33;
        Result[base + 34] = SUM34;
        Result[base + 35] = SUM35;
        Result[base + 36] = SUM36;
        Result[base + 37] = SUM37;
        Result[base + 38] = SUM38;
        Result[base + 39] = SUM39;
        Result[base + 40] = Result[base + 4];
        Result[base + 41] = Result[base + 14];
        Result[base + 42] = Result[base + 24];
        Result[base + 43] = Result[base + 34];
        Result[base + 44] = SUM44;
        Result[base + 45] = SUM45;
        Result[base + 46] = SUM46;
        Result[base + 47] = SUM47;
        Result[base + 48] = SUM48;
        Result[base + 49] = SUM49;
        Result[base + 50] = Result[base + 5];
        Result[base + 51] = Result[base + 15];
        Result[base + 52] = Result[base + 25];
        Result[base + 53] = Result[base + 35];
        Result[base + 54] = Result[base + 45];
        Result[base + 55] = SUM55;
        Result[base + 56] = SUM56;
        Result[base + 57] = SUM57;
        Result[base + 58] = SUM58;
        Result[base + 59] = SUM59;
        Result[base + 60] = Result[base + 6];
        Result[base + 61] = Result[base + 16];
        Result[base + 62] = Result[base + 26];
        Result[base + 63] = Result[base + 36];
        Result[base + 64] = Result[base + 46];
        Result[base + 65] = Result[base + 56];
        Result[base + 66] = SUM66;
        Result[base + 67] = SUM67;
        Result[base + 68] = SUM68;
        Result[base + 69] = SUM69;
        Result[base + 70] = Result[base + 7];
        Result[base + 71] = Result[base + 17];
        Result[base + 72] = Result[base + 27];
        Result[base + 73] = Result[base + 37];
        Result[base + 74] = Result[base + 47];
        Result[base + 75] = Result[base + 57];
        Result[base + 76] = Result[base + 67];
        Result[base + 77] = SUM77;
        Result[base + 78] = SUM78;
        Result[base + 79] = SUM79;
        Result[base + 80] = Result[base + 8];
        Result[base + 81] = Result[base + 18];
        Result[base + 82] = Result[base + 28];
        Result[base + 83] = Result[base + 38];
        Result[base + 84] = Result[base + 48];
        Result[base + 85] = Result[base + 58];
        Result[base + 86] = Result[base + 68];
        Result[base + 87] = Result[base + 78];
        Result[base + 88] = SUM88;
        Result[base + 89] = SUM89;
        Result[base + 90] = Result[base + 9];
        Result[base + 91] = Result[base + 19];
        Result[base + 92] = Result[base + 29];
        Result[base + 93] = Result[base + 39];
        Result[base + 94] = Result[base + 49];
        Result[base + 95] = Result[base + 59];
        Result[base + 96] = Result[base + 69];
        Result[base + 97] = Result[base + 79];
        Result[base + 98] = Result[base + 89];
        Result[base + 99] = SUM99;
    } else {
        for (size_t K = local_id; K < i; K += local_size) {
            offset[K] = idx[ptr + K] * j;
            for (unsigned I = 0; I < j; ++I) {
                a[K * j + I] = H[offset[K] + I];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (unsigned S = 0; S < i; S++) {
            SUM0 += a[S * j] * a[S * j];
            SUM1 += a[S * j] * a[S * j + 1];
            SUM2 += a[S * j] * a[S * j + 2];
            SUM3 += a[S * j] * a[S * j + 3];
            SUM4 += a[S * j] * a[S * j + 4];
            SUM5 += a[S * j] * a[S * j + 5];
            SUM6 += a[S * j] * a[S * j + 6];
            SUM7 += a[S * j] * a[S * j + 7];
            SUM8 += a[S * j] * a[S * j + 8];
            SUM9 += a[S * j] * a[S * j + 9];

            SUM11 += a[S * j + 1] * a[S * j + 1];
            SUM12 += a[S * j + 1] * a[S * j + 2];
            SUM13 += a[S * j + 1] * a[S * j + 3];
            SUM14 += a[S * j + 1] * a[S * j + 4];
            SUM15 += a[S * j + 1] * a[S * j + 5];
            SUM16 += a[S * j + 1] * a[S * j + 6];
            SUM17 += a[S * j + 1] * a[S * j + 7];
            SUM18 += a[S * j + 1] * a[S * j + 8];
            SUM19 += a[S * j + 1] * a[S * j + 9];

            SUM22 += a[S * j + 2] * a[S * j + 2];
            SUM23 += a[S * j + 2] * a[S * j + 3];
            SUM24 += a[S * j + 2] * a[S * j + 4];
            SUM25 += a[S * j + 2] * a[S * j + 5];
            SUM26 += a[S * j + 2] * a[S * j + 6];
            SUM27 += a[S * j + 2] * a[S * j + 7];
            SUM28 += a[S * j + 2] * a[S * j + 8];
            SUM29 += a[S * j + 2] * a[S * j + 9];

            SUM33 += a[S * j + 3] * a[S * j + 3];
            SUM34 += a[S * j + 3] * a[S * j + 4];
            SUM35 += a[S * j + 3] * a[S * j + 5];
            SUM36 += a[S * j + 3] * a[S * j + 6];
            SUM37 += a[S * j + 3] * a[S * j + 7];
            SUM38 += a[S * j + 3] * a[S * j + 8];
            SUM39 += a[S * j + 3] * a[S * j + 9];

            SUM44 += a[S * j + 4] * a[S * j + 4];
            SUM45 += a[S * j + 4] * a[S * j + 5];
            SUM46 += a[S * j + 4] * a[S * j + 6];
            SUM47 += a[S * j + 4] * a[S * j + 7];
            SUM48 += a[S * j + 4] * a[S * j + 8];
            SUM49 += a[S * j + 4] * a[S * j + 9];

            SUM55 += a[S * j + 5] * a[S * j + 5];
            SUM56 += a[S * j + 5] * a[S * j + 6];
            SUM57 += a[S * j + 5] * a[S * j + 7];
            SUM58 += a[S * j + 5] * a[S * j + 8];
            SUM59 += a[S * j + 5] * a[S * j + 9];

            SUM66 += a[S * j + 6] * a[S * j + 6];
            SUM67 += a[S * j + 6] * a[S * j + 7];
            SUM68 += a[S * j + 6] * a[S * j + 8];
            SUM69 += a[S * j + 6] * a[S * j + 9];

            SUM77 += a[S * j + 7] * a[S * j + 7];
            SUM78 += a[S * j + 7] * a[S * j + 8];
            SUM79 += a[S * j + 7] * a[S * j + 9];

            SUM88 += a[S * j + 8] * a[S * j + 8];
            SUM89 += a[S * j + 8] * a[S * j + 9];

            SUM99 += a[S * j + 9] * a[S * j + 9];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        Result[base + 0] = SUM0;
        Result[base + 1] = SUM1;
        Result[base + 2] = SUM2;
        Result[base + 3] = SUM3;
        Result[base + 4] = SUM4;
        Result[base + 5] = SUM5;
        Result[base + 6] = SUM6;
        Result[base + 7] = SUM7;
        Result[base + 8] = SUM8;
        Result[base + 9] = SUM9;
        Result[base + 10] = Result[base + 1];
        Result[base + 11] = SUM11;
        Result[base + 12] = SUM12;
        Result[base + 13] = SUM13;
        Result[base + 14] = SUM14;
        Result[base + 15] = SUM15;
        Result[base + 16] = SUM16;
        Result[base + 17] = SUM17;
        Result[base + 18] = SUM18;
        Result[base + 19] = SUM19;
        Result[base + 20] = Result[base + 2];
        Result[base + 21] = Result[base + 12];
        Result[base + 22] = SUM22;
        Result[base + 23] = SUM23;
        Result[base + 24] = SUM24;
        Result[base + 25] = SUM25;
        Result[base + 26] = SUM26;
        Result[base + 27] = SUM27;
        Result[base + 28] = SUM28;
        Result[base + 29] = SUM29;
        Result[base + 30] = Result[base + 3];
        Result[base + 31] = Result[base + 13];
        Result[base + 32] = Result[base + 23];
        Result[base + 33] = SUM33;
        Result[base + 34] = SUM34;
        Result[base + 35] = SUM35;
        Result[base + 36] = SUM36;
        Result[base + 37] = SUM37;
        Result[base + 38] = SUM38;
        Result[base + 39] = SUM39;
        Result[base + 40] = Result[base + 4];
        Result[base + 41] = Result[base + 14];
        Result[base + 42] = Result[base + 24];
        Result[base + 43] = Result[base + 34];
        Result[base + 44] = SUM44;
        Result[base + 45] = SUM45;
        Result[base + 46] = SUM46;
        Result[base + 47] = SUM47;
        Result[base + 48] = SUM48;
        Result[base + 49] = SUM49;
        Result[base + 50] = Result[base + 5];
        Result[base + 51] = Result[base + 15];
        Result[base + 52] = Result[base + 25];
        Result[base + 53] = Result[base + 35];
        Result[base + 54] = Result[base + 45];
        Result[base + 55] = SUM55;
        Result[base + 56] = SUM56;
        Result[base + 57] = SUM57;
        Result[base + 58] = SUM58;
        Result[base + 59] = SUM59;
        Result[base + 60] = Result[base + 6];
        Result[base + 61] = Result[base + 16];
        Result[base + 62] = Result[base + 26];
        Result[base + 63] = Result[base + 36];
        Result[base + 64] = Result[base + 46];
        Result[base + 65] = Result[base + 56];
        Result[base + 66] = SUM66;
        Result[base + 67] = SUM67;
        Result[base + 68] = SUM68;
        Result[base + 69] = SUM69;
        Result[base + 70] = Result[base + 7];
        Result[base + 71] = Result[base + 17];
        Result[base + 72] = Result[base + 27];
        Result[base + 73] = Result[base + 37];
        Result[base + 74] = Result[base + 47];
        Result[base + 75] = Result[base + 57];
        Result[base + 76] = Result[base + 67];
        Result[base + 77] = SUM77;
        Result[base + 78] = SUM78;
        Result[base + 79] = SUM79;
        Result[base + 80] = Result[base + 8];
        Result[base + 81] = Result[base + 18];
        Result[base + 82] = Result[base + 28];
        Result[base + 83] = Result[base + 38];
        Result[base + 84] = Result[base + 48];
        Result[base + 85] = Result[base + 58];
        Result[base + 86] = Result[base + 68];
        Result[base + 87] = Result[base + 78];
        Result[base + 88] = SUM88;
        Result[base + 89] = SUM89;
        Result[base + 90] = Result[base + 9];
        Result[base + 91] = Result[base + 19];
        Result[base + 92] = Result[base + 29];
        Result[base + 93] = Result[base + 39];
        Result[base + 94] = Result[base + 49];
        Result[base + 95] = Result[base + 59];
        Result[base + 96] = Result[base + 69];
        Result[base + 97] = Result[base + 79];
        Result[base + 98] = Result[base + 89];
        Result[base + 99] = SUM99;
    }
}

__kernel void batchsolve(ulong n, ulong i, ulong j, __global VALUE_TYPE* W, __global VALUE_TYPE* result,
                           __global const VALUE_TYPE* val, __global const unsigned* col_ptr,
                           __global const unsigned* row_idx) {
    size_t basev = get_group_id(0) * j;
    size_t local_id = get_local_id(0);
    size_t local_size = get_local_size(0);

    __local VALUE_TYPE a[300];
    __local VALUE_TYPE b[30];
    VALUE_TYPE subvector0 = 0, subvector1 = 0, subvector2 = 0, subvector3 = 0, subvector4 = 0, subvector5 = 0, subvector6 = 0, subvector7 = 0, subvector8 = 0, subvector9 = 0;

    unsigned long nn = n / 30;
    if (nn > 0) {
        for (unsigned nm = 0; nm < nn; nm++) {
            for (size_t idx = col_ptr[i] + nm * 30 + local_id; idx < (nm + 1) * 30 + col_ptr[i]; idx += local_size) {
                b[idx - (nm * 30) - col_ptr[i]] = val[idx];
                for (unsigned ii = 0; ii < j; ii++) {
                    a[(idx - (nm * 30) - col_ptr[i]) * j + ii] = W[(row_idx[idx] * j) + ii];
                }
            }
            for (unsigned gh = 0; gh < 30; gh++) {
                subvector0 += b[gh] * a[gh * j];
                subvector1 += b[gh] * a[gh * j + 1];
                subvector2 += b[gh] * a[gh * j + 2];
                subvector3 += b[gh] * a[gh * j + 3];
                subvector4 += b[gh] * a[gh * j + 4];
                subvector5 += b[gh] * a[gh * j + 5];
                subvector6 += b[gh] * a[gh * j + 6];
                subvector7 += b[gh] * a[gh * j + 7];
                subvector8 += b[gh] * a[gh * j + 8];
                subvector9 += b[gh] * a[gh * j + 9];
            }
        }
        for (size_t idx = col_ptr[i] + nn * 30 + local_id; idx < col_ptr[i + 1]; idx += local_size) {
            b[idx - (nn * 30) - col_ptr[i]] = val[idx];
            for (unsigned ii = 0; ii < j; ii++) {
                a[(idx - (nn * 30) - col_ptr[i]) * j + ii] = W[(row_idx[idx] * j) + ii];
            }
        }
        for (unsigned gh = 0; gh < col_ptr[i + 1] - col_ptr[i] - nn * 30; gh++) {
            subvector0 += b[gh] * a[gh * j];
            subvector1 += b[gh] * a[gh * j + 1];
            subvector2 += b[gh] * a[gh * j + 2];
            subvector3 += b[gh] * a[gh * j + 3];
            subvector4 += b[gh] * a[gh * j + 4];
            subvector5 += b[gh] * a[gh * j + 5];
            subvector6 += b[gh] * a[gh * j + 6];
            subvector7 += b[gh] * a[gh * j + 7];
            subvector8 += b[gh] * a[gh * j + 8];
            subvector9 += b[gh] * a[gh * j + 9];
        }
    } else {
        for (size_t idx = col_ptr[i] + local_id; idx < col_ptr[i + 1]; idx += local_size) {
            b[idx - col_ptr[i]] = val[idx];
            for (unsigned ii = 0; ii < j; ii++) {
                a[(idx - col_ptr[i]) * j + ii] = W[(row_idx[idx] * j) + ii];
            }
        }
        for (unsigned gh = 0; gh < n; gh++) {
            subvector0 += b[gh] * a[gh * j];
            subvector1 += b[gh] * a[gh * j + 1];
            subvector2 += b[gh] * a[gh * j + 2];
            subvector3 += b[gh] * a[gh * j + 3];
            subvector4 += b[gh] * a[gh * j + 4];
            subvector5 += b[gh] * a[gh * j + 5];
            subvector6 += b[gh] * a[gh * j + 6];
            subvector7 += b[gh] * a[gh * j + 7];
            subvector8 += b[gh] * a[gh * j + 8];
            subvector9 += b[gh] * a[gh * j + 9];
        }
    }
    result[basev + 0] = subvector0;
    result[basev + 1] = subvector1;
    result[basev + 2] = subvector2;
    result[basev + 3] = subvector3;
    result[basev + 4] = subvector4;
    result[basev + 5] = subvector5;
    result[basev + 6] = subvector6;
    result[basev + 7] = subvector7;
    result[basev + 8] = subvector8;
    result[basev + 9] = subvector9;
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
                                   __global VALUE_TYPE* subMatrix) {
    size_t local_id = get_local_id(0);
    size_t local_size = get_local_size(0);
    size_t group_id = get_group_id(0);
    size_t num_groups = get_num_groups(0);
    size_t base = group_id * k * k;
    size_t baseV = group_id * k;

    for (size_t Rw = group_id; Rw < rows; Rw += num_groups) {
        __global VALUE_TYPE* Wr = &W[Rw * k];
        unsigned omegaSize = row_ptr[Rw + 1] - row_ptr[Rw];

        if (omegaSize > 0) {
            Mt_byM_multiply_k(omegaSize, k, H, subMatrix, row_ptr[Rw], col_idx);

            barrier(CLK_GLOBAL_MEM_FENCE);

            for (size_t c = local_id; c < k; c += local_size) {
                subMatrix[base + c * k + c] += lambda;
            }

            barrier(CLK_GLOBAL_MEM_FENCE);

            if (local_id == 0) {
                inverseMatrix_CholeskyMethod(k, subMatrix, p);
            }

            barrier(CLK_GLOBAL_MEM_FENCE);



            batchsolve(omegaSize, Rw, k, H, subVector, val_t, row_ptr, col_idx);



            barrier(CLK_GLOBAL_MEM_FENCE);

            for (size_t c = local_id; c < k; c += local_size) {
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
    size_t local_id = get_local_id(0);
    size_t local_size = get_local_size(0);
    size_t group_id = get_group_id(0);
    size_t num_groups = get_num_groups(0);
    size_t base = group_id * k * k;
    size_t baseV = group_id * k;

    for (size_t Rh = group_id; Rh < cols; Rh += num_groups) {
        __global VALUE_TYPE* Hr = &H[Rh * k];
        unsigned omegaSize = col_ptr[Rh + 1] - col_ptr[Rh];

        if (omegaSize > 0) {
            Mt_byM_multiply_k(omegaSize, k, W, subMatrix, col_ptr[Rh], row_idx);

            barrier(CLK_GLOBAL_MEM_FENCE);

            for (size_t c = local_id; c < k; c += local_size) {
                subMatrix[base + c * k + c] += lambda;
            }

            barrier(CLK_GLOBAL_MEM_FENCE);

            if (local_id == 0) {
                inverseMatrix_CholeskyMethod(k, subMatrix, p);
            }

            barrier(CLK_GLOBAL_MEM_FENCE);



            batchsolve(omegaSize, Rh, k, W, subVector, val, col_ptr, row_idx);



            barrier(CLK_GLOBAL_MEM_FENCE);

            for (size_t c = local_id; c < k; c += local_size) {
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
