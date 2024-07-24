#include "../test_hmatrix_build.hpp"

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    bool is_error = false;

    for (auto nr : {200, 400}) {
        for (auto nc : {200, 400}) {
            for (auto use_local_cluster : {true, false}) {
                for (auto epsilon : {1e-14, 1e-6}) {
                    for (auto use_dense_block_generator : {true, false}) {
                        std::cout << nr << " " << nc << " " << use_local_cluster << " " << epsilon << " " << use_dense_block_generator << "\n";

                        is_error = is_error || test_hmatrix_build<std::complex<double>, GeneratorTestComplexSymmetric>(nr, nc, use_local_cluster, 'N', 'N', epsilon, use_dense_block_generator);
                        if (nr == nc) {
                            for (auto UPLO : {'U', 'L'}) {
                                is_error = is_error || test_hmatrix_build<std::complex<double>, GeneratorTestComplexSymmetric>(nr, nr, use_local_cluster, 'S', UPLO, epsilon, use_dense_block_generator);

                                is_error = is_error || test_hmatrix_build<std::complex<double>, GeneratorTestComplexHermitian>(nr, nr, use_local_cluster, 'H', UPLO, epsilon, use_dense_block_generator);
                            }
                        }
                    }
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
