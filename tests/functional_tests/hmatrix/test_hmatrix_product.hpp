#include "test_hmatrix_hmatrix_product.hpp"
#include "test_hmatrix_lrmat_product.hpp"
#include "test_hmatrix_matrix_product.hpp"
#include "test_lrmat_hmatrix_product.hpp"
#include "test_matrix_hmatrix_product.hpp"
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
bool test_hmatrix_product(char transa, char transb, int n1, int n2, int n3, bool use_local_cluster, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    // Get the number of processes
    int sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);

    // Get the rankWorld of the process
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    bool is_error = false;
    TestCaseProduct<T, GeneratorTestType> test_case(transa, transb, n1, n2, n3, 1, 2, sizeWorld);

    is_error = test_hmatrix_matrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, margin);
    is_error = test_matrix_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, margin);
    is_error = test_hmatrix_lrmat_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, margin);
    is_error = test_lrmat_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, margin);
    is_error = test_hmatrix_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, margin);
    return is_error;
}

template <typename T, typename GeneratorTestType>
bool test_symmetric_hmatrix_product(int n1, int n2, char side, char UPLO, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    // Get the number of processes
    int sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);

    // Get the rankWorld of the process
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    bool is_error = false;
    TestCaseSymmetricProduct<T, GeneratorTestType> test_case(n1, n2, 2, side, 'S', UPLO, sizeWorld);

    is_error = test_symmetric_hmatrix_matrix_product<T, GeneratorTestType>(test_case, epsilon, margin);
    // is_error = test_symmetric_hmatrix_lrmat_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon);
    // is_error = test_symmetric_hmatrix_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, margin);
    // if (!(side == 'L' && Symmetry != 'N')) {
    // is_error = test_lrmat_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon);
    // }
    return is_error;
}
