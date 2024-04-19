#include "../test_hmatrix_product.hpp"
#include <htool/hmatrix/hmatrix.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    bool is_error       = false;
    const double margin = 10;

    for (auto epsilon : {1e-6, 1e-10}) {
        for (auto n1 : {200, 400}) {
            for (auto n3 : {100}) {
                for (auto use_local_cluster : {true, false}) {
                    for (auto n2 : {200, 400}) {
                        for (auto transa : {'N', 'T'}) {
                            for (auto transb : {'N', 'T'}) {
                                std::cout << "hmatrix product: " << use_local_cluster << " " << epsilon << " " << n1 << " " << n2 << " " << n3 << " " << transa << " " << transb << "\n";

                                is_error = is_error || test_hmatrix_product<double, GeneratorTestDouble>(transa, transb, n1, n2, n3, use_local_cluster, epsilon, margin);
                            }
                        }
                    }
                }

                // TODO: fix 'C' operation, missing some conj operations somewhere
                // for (auto operation : {'N', 'C'}) {

                //     // Square matrix
                //     is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexSymmetric>(operation, 'N', n1, n2, n3, 'N', 'N', 'N', use_local_cluster, epsilon, margin);
                //     is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexHermitian>(operation, 'N', n1, n2, n3, 'L', 'H', 'L', use_local_cluster, epsilon, margin);
                //     is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexHermitian>(operation, 'N', n1, n2, n3, 'L', 'H', 'U', use_local_cluster, epsilon, margin);

                //     // Rectangle matrix
                //     is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplex>(operation, 'N', n1_increased, n2, n3, 'N', 'N', 'N', use_local_cluster, epsilon, margin);
                //     is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplex>(operation, 'N', n1, n2_increased, n3, 'N', 'N', 'N', use_local_cluster, epsilon, margin);
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
