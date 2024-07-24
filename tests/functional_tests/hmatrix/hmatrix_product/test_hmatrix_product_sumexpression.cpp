#include "../test_hmatrix_product.hpp"
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/linalg/interface.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/testing/generate_test_case.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_test(int n1, int n2, int n3, bool use_local_cluster, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    // Get the number of processes
    int sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);

    // Get the rankWorld of the process
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    bool is_error = false;
    TestCaseProduct<T, GeneratorTestType> test_case('N', 'N', n1, n2, n3, 1, 2, sizeWorld);

    is_error = test_hmatrix_hmatrix_product_sumexpression<T, GeneratorTestType>(test_case, epsilon, margin, use_local_cluster);
    return is_error;
}
int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    bool is_error       = false;
    const double margin = 10;

    for (auto epsilon : {1e-6, 1e-10}) {
        for (auto n1 : {200, 400}) {
            for (auto n3 : {200, 400}) {
                for (auto use_local_cluster : {true}) {
                    for (auto n2 : {200, 400}) {

                        is_error = test_test<double, GeneratorTestDouble>(n1, n2, n3, use_local_cluster, epsilon, margin);
                    }
                }
                // for (auto side : {'L', 'R'}) {
                //     for (auto UPLO : {'U', 'L'}) {
                //         std::cout << "symmetric matrix product: " << n1 << " " << n3 << " " << side << " " << UPLO << " " << epsilon << "\n";
                //         is_error = is_error || test_symmetric_hmatrix_product<double, GeneratorTestDoubleSymmetric>(n1, n3, side, UPLO, epsilon, margin);
                //     }
                // }
            }
        }
    }
    MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}
