#include <chrono>

#include "tools.h"
#include "extra.h"
#include "clmf.h"

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
    bool with_weights = false;
    bool ifALS = true;
    load(param.src_dir, R, ifALS, with_weights);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT34 = t4 - t3;
    std::cout << "[INFO] Loading rating data time: " << deltaT34.count() << "s.\n";
    std::cout << "------------------------------------------------------" << std::endl;

    mat_t W_c;
    mat_t H_c;
    initial_col(W_c, R.rows, param.k);
    initial_col(H_c, R.cols, param.k);

    mat_t W_ref;
    mat_t H_ref;
    initial_col(W_ref, R.rows, param.k);
    initial_col(H_ref, R.cols, param.k);

    doit(R, W_c, H_c, param, param.opencl_filename);

    std::chrono::duration<double> deltaT56{};
    std::chrono::duration<double> deltaT9_10{};

    // Predict RMSE with the W and H matrices produced by OpenCL kernels
    if (param.do_predict == 1) {
        std::cout << "------------------------------------------------------" << std::endl;
        auto t5 = std::chrono::high_resolution_clock::now();
        calculate_rmse(W_c, H_c, param.src_dir, param.k);
        auto t6 = std::chrono::high_resolution_clock::now();
        deltaT56 = t6 - t5;
        std::cout << "[info] Predict Time: " << deltaT56.count() << " s.\n";
    }

    // Compare OpenCL results with reference OpenMP results
    if (param.do_ref == 1) {
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "[info] Computing clMF OpenMP reference results on CPU." << std::endl;
        auto t9 = std::chrono::high_resolution_clock::now();
        ALS_multicore(R, W_ref, H_ref, param);
        auto t10 = std::chrono::high_resolution_clock::now();
        deltaT9_10 = t10 - t9;
        std::cout << "[info] OMP Predict Time: " << deltaT9_10.count() << " s.\n";
        std::cout << "[info] validate the results." << std::endl;
        golden_compare(W_c, W_ref, R.rows, param.k);
        golden_compare(H_c, H_ref, R.cols, param.k);
        calculate_rmse(W_ref, H_ref, param.src_dir, param.k);
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
              << " s.\n";
    return EXIT_SUCCESS;
}
