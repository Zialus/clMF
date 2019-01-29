#include "tools.h"
#include "clmf_ref.h"
#include "clmf_ocl.h"

std::chrono::duration<double> deltaT12;
std::chrono::duration<double> deltaTAB;

int main(int argc, char* argv[]) {
    auto t7 = std::chrono::high_resolution_clock::now();

    parameter param = parse_command_line(argc, argv);

    if (param.verbose) {
        print_all_the_info();
    }

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "[info] Loading R matrix..." << std::endl;
    auto t3 = std::chrono::high_resolution_clock::now();
    smat_t R;
    testset_t T;
    bool ifALS = true;
    load(param.src_dir, R, T, ifALS);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT34 = t4 - t3;
    std::cout << "[info] Loading rating data time: " << deltaT34.count() << "s.\n";
    std::cout << "------------------------------------------------------" << std::endl;

    mat_t W_c;
    mat_t H_c;
    initial_col(W_c, R.rows, param.k);
    initial_col(H_c, R.cols, param.k);

    mat_t W_ref;
    mat_t H_ref;
    initial_col(W_ref, R.rows, param.k);
    initial_col(H_ref, R.cols, param.k);

    switch (param.version) {
        case 1: {
            std::cout << "[info] Picked Version 1: ALS rolled" << std::endl;
            char kcode_filename[1024 + 15];
            snprintf(kcode_filename, sizeof(kcode_filename), "%s/ALS_rolled.cl", param.kcode_path);
            clmf(R, W_c, H_c, T, param, kcode_filename);
            break;
        }
        case 2: {
            std::cout << "[info] Picked Version 2: ALS unrolled" << std::endl;
            char kcode_filename[1024 + 15];
            snprintf(kcode_filename, sizeof(kcode_filename), "%s/ALS.cl", param.kcode_path);
            clmf(R, W_c, H_c, T, param, kcode_filename);
            break;
        }
        default: {
            printf("[FAILED] Wrong version");
            return EXIT_FAILURE;
        }
    }

    std::chrono::duration<double> deltaT56{};
    std::chrono::duration<double> deltaT9_10{};
    std::chrono::duration<double> deltaT11_12{};
    std::chrono::duration<double> deltaT13_14{};

    // Predict RMSE with the W and H matrices produced by OpenCL kernels
    if (param.do_predict == 1) {
        std::cout << "------------------------------------------------------" << std::endl;
        auto t5 = std::chrono::high_resolution_clock::now();
        calculate_rmse(W_c, H_c, param.src_dir, param.k);
        auto t6 = std::chrono::high_resolution_clock::now();
        deltaT56 = t6 - t5;
        std::cout << "[info] OCL Predict Time: " << deltaT56.count() << " s.\n";
    }

    // Compare OpenCL results with reference OpenMP results
    if (param.do_ref == 1) {
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "[info] Computing clMF OpenMP reference results on CPU." << std::endl;
        auto t9 = std::chrono::high_resolution_clock::now();
        clmf_ref(R, W_ref, H_ref, T, param);
        auto t10 = std::chrono::high_resolution_clock::now();
        deltaT9_10 = t10 - t9;
        std::cout << "[info] OMP Training Time: " << deltaT9_10.count() << " s.\n";

        std::cout << "------------------------------------------------------" << std::endl;
        auto t13 = std::chrono::high_resolution_clock::now();
        calculate_rmse(W_ref, H_ref, param.src_dir, param.k);
        auto t14 = std::chrono::high_resolution_clock::now();
        deltaT13_14 = t14 - t13;
        std::cout << "[info] OMP Predict Time: " << deltaT13_14.count() << " s.\n";

        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "[info] validate the results." << std::endl;
        auto t11 = std::chrono::high_resolution_clock::now();
        golden_compare(W_c, W_ref, R.rows, param.k);
        golden_compare(H_c, H_ref, R.cols, param.k);
        auto t12 = std::chrono::high_resolution_clock::now();
        deltaT11_12 = t12 - t11;
        std::cout << "[info] Validate Time: " << deltaT11_12.count() << " s.\n";
    }
    std::cout << "------------------------------------------------------" << std::endl;

    // Some print debugging
//    print_matrix(W_c, R.rows, param.k);
//    print_matrix(H_c, R.cols, param.k);
//
//    print_matrix(W_ref, R.rows, param.k);
//    print_matrix(H_ref, R.cols, param.k);

    auto t8 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT78 = t8 - t7;
    std::cout << "Total Time: " << deltaT78.count() << " Parcial Sums:"
              << deltaT12.count() + deltaT34.count() + deltaT56.count() + deltaTAB.count() + deltaT9_10.count()
                 + deltaT11_12.count() + deltaT13_14.count() << " s.\n";
    return EXIT_SUCCESS;
}
