#ifndef HTOOL_MATRIX_LINALG_FACTORIZATION_HPP
#define HTOOL_MATRIX_LINALG_FACTORIZATION_HPP

#include "../../wrappers/wrapper_lapack.hpp"
#include "../matrix.hpp"
namespace htool {

template <typename T>
void lu_factorization(Matrix<T> &A) {
    int M      = A.nb_rows();
    int N      = A.nb_cols();
    int lda    = M;
    auto &ipiv = A.get_pivots();
    ipiv.resize(std::min(M, N));
    int info;

    Lapack<T>::getrf(&M, &N, A.data(), &lda, ipiv.data(), &info);
}

template <typename T>
void triangular_matrix_matrix_solve(char side, char UPLO, char transa, char diag, T alpha, const Matrix<T> &A, Matrix<T> &B) {
    int m   = B.nb_rows();
    int n   = B.nb_cols();
    int lda = side == 'L' ? m : n;
    int ldb = m;

    Blas<T>::trsm(&side, &UPLO, &transa, &diag, &m, &n, &alpha, A.data(), &lda, B.data(), &ldb);
}
template <typename T>
void triangular_matrix_vec_solve(char side, char UPLO, char transa, char diag, T alpha, const Matrix<T> &A, std::vector<T> &B) {
    int m   = B.size();
    int n   = 1;
    int lda = side == 'L' ? m : n;
    int ldb = m;

    Blas<T>::trsm(&side, &UPLO, &transa, &diag, &m, &n, &alpha, A.data(), &lda, B.data(), &ldb);
}

template <typename T>
void lu_solve(char trans, const Matrix<T> &A, Matrix<T> &B) {
    int M      = A.nb_rows();
    int NRHS   = B.nb_cols();
    int lda    = M;
    int ldb    = M;
    auto &ipiv = A.get_pivots();
    int info;

    Lapack<T>::getrs(&trans, &M, &NRHS, A.data(), &lda, ipiv.data(), B.data(), &ldb, &info);
}

template <typename T>
void cholesky_factorization(char UPLO, Matrix<T> &A) {
    int M   = A.nb_rows();
    int lda = M;
    int info;

    Lapack<T>::potrf(&UPLO, &M, A.data(), &lda, &info);
}

template <typename T>
void cholesky_solve(char UPLO, const Matrix<T> &A, Matrix<T> &B) {
    int M    = A.nb_rows();
    int NRHS = B.nb_cols();
    int lda  = M;
    int ldb  = M;
    int info;

    Lapack<T>::potrs(&UPLO, &M, &NRHS, A.data(), &lda, B.data(), &ldb, &info);
}

template <typename T>
void get_lu_factorisation(const Matrix<T> &M, Matrix<T> &L, Matrix<T> &U, std::vector<int> &P) {
    auto A   = M;
    int size = A.nb_rows();
    std::vector<int> ipiv(size, 0.0);
    int info = -1;
    Lapack<T>::getrf(&size, &size, A.data(), &size, ipiv.data(), &info);
    for (int i = 0; i < size; ++i) {
        L(i, i) = 1;
        U(i, i) = A(i, i);

        for (int j = 0; j < i; ++j) {
            L(i, j) = A(i, j);
            U(j, i) = A(j, i);
        }
    }
    for (int k = 1; k < size + 1; ++k) {
        P[k - 1] = k;
    }
    for (int k = 0; k < size; ++k) {
        if (ipiv[k] - 1 != k) {
            int temp       = P[k];
            P[k]           = P[ipiv[k] - 1];
            P[ipiv[k] - 1] = temp;
        }
    }
}
/// A =SVD -> lapack ------------> je le fait pour renvoyer S et écrire U et V sur des matrices d'entrées !!!!! MARCHE QUE AVEC DES MATRICES CARRE sinon il faut changer lda, ldb etc
template <typename T>
std::vector<T> compute_svd(Matrix<T> &A, Matrix<T> &U, Matrix<T> &V) {
    int N = A.nb_cols();
    std::vector<T> S(N, 0.0);
    int info;
    int lwork = 10 * N;
    std::vector<T> rwork(lwork);
    std::vector<T> work(lwork);
    Lapack<T>::gesvd("A", "A", &N, &N, A.data(), &N, S.data(), U.data(), &N, V.data(), &N, work.data(), &lwork, rwork.data(), &info);
    return S;
}

} // namespace htool
#endif
