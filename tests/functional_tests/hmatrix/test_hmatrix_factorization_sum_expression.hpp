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
bool test_hlu_sum_expression(int n1, int n2, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(*test_case.root_cluster_A_output, *test_case.root_cluster_A_input, epsilon, eta, 'N', 'N', -1, -1, -1);
    HMatrix<T, htool::underlying_type<T>> A = hmatrix_tree_builder_A.build(*test_case.operator_A);

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A);
    std::vector<T> B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;

    test_case.operator_A->copy_submatrix(no_A, ni_A, test_case.root_cluster_A_output->get_offset(), test_case.root_cluster_A_input->get_offset(), A_dense.data());
    auto X_dense = generate_random_vector(no_X);

    // generate_random_matrix(X_dense);
    // add_matrix_matrix_product(trans, 'N', T(1.), A_dense, X_dense, T(0.), B_dense);
    B_dense = A_dense * X_dense;

    // LU factorization
    matrix_test = B_dense;
    // lu_factorization(A);
    // lu_solve(trans, A, matrix_test);
    HMatrix<T, htool::underlying_type<T>> Lres(A.get_target_cluster(), A.get_source_cluster());
    HMatrix<T, htool::underlying_type<T>> Ures(A.get_target_cluster(), A.get_source_cluster());
    Lres.set_admissibility_condition(A.get_admissibility_condition());
    Ures.set_admissibility_condition(A.get_admissibility_condition());
    Lres.set_eta(eta);
    Ures.set_eta(eta);
    Lres.set_low_rank_generator(A.get_low_rank_generator());
    Ures.set_low_rank_generator(A.get_low_rank_generator());
    Lres.set_epsilon(epsilon);
    Ures.set_epsilon(epsilon);
    HLU_fast(A, A.get_target_cluster(), &Lres, &Ures);

    auto ytest = Lres.solve_LU_triangular(Lres, Ures, matrix_test);
    error      = norm2(X_dense - ytest) / norm2(X_dense);

    // error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix lu solve: " << error << endl;
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}
