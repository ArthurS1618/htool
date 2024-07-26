#include "../test_hmatrix_factorization_sum_expression.hpp"
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/testing/generate_test_case.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    // MPI_Init(&argc, &argv);
    bool is_error       = false;
    const int n1        = 10000;
    const double margin = 1;
    const int n2        = 100;
    std::cout << "size : " << n1 << std::endl;
    for (auto epsilon : {1e-3, 1e-6}) {
        is_error = is_error || test_hlu_sum_expression<double, GeneratorTestDoubleSymmetric>(n1, n2, epsilon, margin);

        // for (auto UPLO : {'L', 'U'}) {
        //     is_error = is_error || test_hmatrix_cholesky<double, GeneratorTestDoubleSymmetric>(UPLO, n1, n2, epsilon, margin);
        // }
    }
    // MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}
