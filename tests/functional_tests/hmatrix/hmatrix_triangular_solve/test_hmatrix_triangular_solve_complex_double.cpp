#include "../test_hmatrix_triangular_solve.hpp"
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/testing/generate_test_case.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    bool is_error       = false;
    const int n1        = 400;
    const double margin = 1;

    for (auto number_of_rhs : {100}) {
        for (auto epsilon : {1e-6, 1e-10}) {
            for (auto side : {'L', 'R'}) {
                // TODO: add 'C' operation when hmatrix product is checked
                for (auto operation : {'N', 'T'}) {
                    std::cout << epsilon << " " << number_of_rhs << " " << side << " " << operation << "\n";
                    is_error = is_error || test_hmatrix_triangular_solve<std::complex<double>, GeneratorTestComplexHermitian>(side, operation, n1, number_of_rhs, epsilon, margin);
                }
            }
        }
    }
    MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}
