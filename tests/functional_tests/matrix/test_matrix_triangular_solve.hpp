#include <htool/matrix/linalg/interface.hpp>
#include <htool/matrix/matrix.hpp>
#include <htool/testing/generator_input.hpp>

using namespace std;
using namespace htool;

template <typename T>
bool test_matrix_triangular_solve(int n, int nrhs, char side, char transa) {

    bool is_error = false;

    // Generate random matrix
    htool::underlying_type<T> error;
    T alpha;
    Matrix<T> result(n, nrhs), B(n, nrhs);
    if (side == 'R') {
        result.resize(nrhs, n);
        B.resize(nrhs, n);
    }
    generate_random_array(result.data(), result.nb_rows() * result.nb_cols());
    generate_random_scalar(alpha);

    Matrix<T> A(n, n), test_factorization, test_solve;
    generate_random_array(A.data(), A.nb_rows() * A.nb_cols());
    for (int i = 0; i < n; i++) {
        T sum = 0;
        for (int j = 0; j < n; j++) {
            sum += std::abs(A(i, j));
        }
        A(i, i) = sum;
    }

    // Triangular setup
    Matrix<T> LA(A), UA(A), LB(B.nb_rows(), B.nb_cols()), UB(B.nb_rows(), B.nb_cols());
    for (int i = 0; i < A.nb_rows(); i++) {
        for (int j = 0; j < A.nb_cols(); j++) {
            if (i > j) {
                UA(i, j) = 0;
            }
            if (i < j) {
                LA(i, j) = 0;
            }
        }
    }

    if (side == 'L') {
        add_matrix_matrix_product(transa, 'N', T(1) / alpha, UA, result, T(0), UB);
        add_matrix_matrix_product(transa, 'N', T(1) / alpha, LA, result, T(0), LB);
    } else if (side == 'R') {
        add_matrix_matrix_product('N', transa, T(1) / alpha, result, UA, T(0), UB);
        add_matrix_matrix_product('N', transa, T(1) / alpha, result, LA, T(0), LB);
    }

    // triangular_matrix_matrix_solve
    test_factorization = LA;
    test_solve         = LB;
    triangular_matrix_matrix_solve(side, 'L', transa, 'N', alpha, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on lower triangular matrix matrix solve: " << error << endl;

    test_factorization = UA;
    test_solve         = UB;
    triangular_matrix_matrix_solve(side, 'U', transa, 'N', alpha, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on upper triangular matrix matrix solve: " << error << endl;

    return is_error;
}
