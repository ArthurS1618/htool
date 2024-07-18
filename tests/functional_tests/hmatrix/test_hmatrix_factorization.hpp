#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/linalg/interface.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/matrix/linalg/interface.hpp>
#include <htool/testing/generate_test_case.hpp>
#include <htool/testing/generator_input.hpp>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_hmatrix_lu(char trans, int n1, int n2, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', trans, n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(*test_case.root_cluster_A_output, *test_case.root_cluster_A_input, epsilon, eta, 'N', 'N', -1, -1, -1);
    HMatrix<T, htool::underlying_type<T>> A = hmatrix_tree_builder_A.build(*test_case.operator_A);

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;
    test_case.operator_A->copy_submatrix(no_A, ni_A, test_case.root_cluster_A_output->get_offset(), test_case.root_cluster_A_input->get_offset(), A_dense.data());
    generate_random_matrix(X_dense);
    add_matrix_matrix_product(trans, 'N', T(1.), A_dense, X_dense, T(0.), B_dense);

    // LU factorization
    matrix_test = B_dense;
    lu_factorization(A);
    lu_solve(trans, A, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix lu solve: " << error << endl;
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}

template <typename T, typename GeneratorTestType>
bool test_hmatrix_cholesky(char UPLO, int n1, int n2, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    bool is_error = false;
    double eta    = -100;
    htool::underlying_type<T> error;

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(*test_case.root_cluster_A_output, *test_case.root_cluster_A_input, epsilon, eta, is_complex<T>() ? 'H' : 'S', UPLO, -1, -1, -1);
    HMatrix<T, htool::underlying_type<T>> A = hmatrix_tree_builder_A.build(*test_case.operator_A);

    save_leaves_with_rank(A, "A");

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;
    test_case.operator_A->copy_submatrix(no_A, ni_A, test_case.root_cluster_A_output->get_offset(), test_case.root_cluster_A_input->get_offset(), A_dense.data());
    generate_random_matrix(X_dense);
    // add_matrix_matrix_product('N', 'N', T(1.), A_dense, X_dense, T(0.), B_dense);
    add_symmetric_matrix_matrix_product('L', UPLO, T(1.), A_dense, X_dense, T(0.), B_dense);

    // LU factorization
    matrix_test = B_dense;
    cholesky_factorization(UPLO, A);
    save_leaves_with_rank(A, "A_chol");
    Matrix<T> A_chol_dense(A.nb_rows(), A.nb_cols());
    copy_to_dense(A, A_chol_dense.data());
    std::ofstream A_chol_file("A_chol_dense");
    A_chol_dense.print(A_chol_file, ",");
    cholesky_solve(UPLO, A, matrix_test);
    std::ofstream matrix_test_file("matrix_test");
    matrix_test.print(matrix_test_file, ",");
    std::ofstream X_dense_file("X_dense");
    X_dense.print(X_dense_file, ",");
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix cholesky solve: " << error << endl;
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}
