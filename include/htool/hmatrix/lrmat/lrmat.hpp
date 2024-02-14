#ifndef HTOOL_LRMAT_HPP
#define HTOOL_LRMAT_HPP

#include "../../basic_types/matrix.hpp"
#include "../../clustering/cluster_node.hpp"
#include "../interfaces/virtual_generator.hpp"
#include "../interfaces/virtual_lrmat_generator.hpp"
#include <cassert>
#include <vector>

namespace htool {

template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
class LowRankMatrix {

  protected:
    // Data member
    int m_rank;
    int m_number_of_rows, m_number_of_columns;
    Matrix<CoefficientPrecision> m_U, m_V;
    underlying_type<CoefficientPrecision> m_epsilon;

  public:
    // Constructors
    LowRankMatrix(const VirtualGenerator<CoefficientPrecision> &A, const VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision> &LRGenerator, const Cluster<CoordinatesPrecision> &target_cluster, const Cluster<CoordinatesPrecision> &source_cluster, int rank = -1, underlying_type<CoefficientPrecision> epsilon = 1e-3) : m_rank(rank), m_number_of_rows(target_cluster.get_size()), m_number_of_columns(source_cluster.get_size()), m_U(), m_V(), m_epsilon(epsilon) {

        if (m_rank == 0) {

            m_U.resize(m_number_of_rows, 1);
            m_V.resize(1, m_number_of_columns);
            std::fill_n(m_U.data(), m_number_of_rows, 0);
            std::fill_n(m_V.data(), m_number_of_columns, 0);
        } else {
            LRGenerator.copy_low_rank_approximation(A, target_cluster, source_cluster, epsilon, m_rank, m_U, m_V);
        }
    }
    // Je rajoute ce constructeur ppour moi -> on lui donne la décomposition UV
    LowRankMatrix(const Matrix<CoefficientPrecision> &U, const Matrix<CoefficientPrecision> &V) : m_rank(U.nb_cols()), m_number_of_rows(U.nb_rows()), m_number_of_columns(V.nb_cols()), m_U(U), m_V(V) {}
    // Getters
    int nb_rows() const { return m_number_of_rows; }
    int nb_cols() const { return m_number_of_columns; }
    int rank_of() const { return m_rank; }

    CoefficientPrecision get_U(int i, int j) const { return m_U(i, j); }
    CoefficientPrecision get_V(int i, int j) const { return m_V(i, j); }
    // Je rajoute ca pour avoir U et V en entier
    ///////////////////////////////////////////
    const Matrix<CoefficientPrecision> Get_U() const { return m_U; }
    const Matrix<CoefficientPrecision> Get_V() const { return m_V; }
    ///////////////////////////////////////////
    void assign_U(int i, int j, CoefficientPrecision *ptr) { return m_U.assign(i, j, ptr); }
    void assign_V(int i, int j, CoefficientPrecision *ptr) { return m_V.assign(i, j, ptr); }
    underlying_type<CoefficientPrecision> get_epsilon() const { return m_epsilon; }

    std::vector<CoefficientPrecision> operator*(const std::vector<CoefficientPrecision> &a) const {
        return m_U * (m_V * a);
    }
    //////////////////////////
    /// je rajoute l'addition tronqué
    ///////////////////
    // est ce qu'il y a un moyen simple de concatener ?
    // void formatted_addition(const Matrix<CoefficientPrecision> &U, const Matrix<CoefficientPrecision> &V) {
    //     Matrix<CoefficientPrecision> Uconc(U.nb_rows(), U.nb_cols() + this->rank_of());
    //     Matrix<CoefficientPrecision> Vconc(V.nb_rows() + this->rank_of(), V.nb_cols());
    //     for (int k = 0; k < U.nb_rows(); ++k) {
    //         for (int l = 0; l < this->rank_of(); ++k) {
    //             Uconc(k, l) = m_U(k, l);
    //         }
    //         for (int l = 0; l < U.nb_cols(); ++l) {
    //             Uconc(k, l + this->rank_of() + l);
    //         }
    //     }
    //     for (int l = 0; l < V.nb_cols(); ++l) {
    //         for (int k = 0; k < this->rank_of(); ++k) {
    //             Vconc(k, l) = m_V(k, l);
    //         }
    //         for (int k = 0; k < V.nb_rows(); ++k) {
    //             Vconc(k, l) = V(k, l);
    //         }
    //     }
    //     std::vector<CoefficientPrecision> Tau_U(this->rank_of() * U.nb_rows());
    //     std::vector<CoefficientPrecision> Tau_V(this->rank_of() * V.nb_rows());
    //     std::vector<double> lworku(this->rank_of() * (U.nb_cols() + this->rank_of()));
    //     std::vector<double> lworkv(this->rank_of() * V.nb_cols());
    //     std::vector<double> worku(this->rank_of() * (U.nb_cols() + this->rank_of()));
    //     std::vector<double> workv(this->rank_of() * V.nb_cols());
    //     int infou, infov;
    //     Lapack<CoefficientPrecision>::geqrf(U.nb_rows(), U.nb_cols() + this->rank_of(), Uconc.data(), U.nb_rows(), Tau_U, worku, lworqu, infou);
    //     Lapack<CoefficientPrecision>::geqrf(V.nb_rows() + this->rank_of(), V.nb_cols(), Vconc.data(), V.nb_cols(), Tau_V, workv, lworqv, infov);
    // }
    // void actualise(const Matrix<CoefficientPrecision> &u, const Matrix<CoefficientPrecision> &v, VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision> &LRGenerator, const Cluster<CoordinatesPrecision> &t, const Cluster<CoordinatesPrecision> &s) {
    //     if (u.nb_rows() == m_U.nb_rows() and v.nb_cols() == m_V.nb_cols()) {

    //         // Matrix<CoefficientPrecision> Uconc(u.nb_rows(), m_U.nb_cols() + u.nb_cols());
    //         // Matrix<CoefficientPrecision> Vconc(m_V.nb_rows() + v.nb_rows(), v.nb_cols());
    //         // // U= (U1, U2) , V= (V1, V2) ;
    //         // for (int k = 0; k < u.nb_rows(); ++k) {
    //         //     for (int l = 0; l < this->Get_U().nb_cols(); ++l) {
    //         //         Uconc(k, l) = m_U(k, l);
    //         //     }
    //         //     for (int l = m_U.nb_cols(); l < m_U.nb_cols() + u.nb_cols(); ++l) {
    //         //         Uconc(k, l) = u(k, l - m_U.nb_cols());
    //         //     }
    //         // }
    //         // for (int l = 0; l < v.nb_cols(); ++l) {
    //         //     for (int k = 0; k < m_V.nb_rows(); ++k) {
    //         //         Vconc(k, l) = m_V(k, l);
    //         //     }
    //         //     for (int k = m_V.nb_rows(); k < m_V.nb_rows() + v.nb_rows(); ++k) {
    //         //         Vconc(k, l) = v(k - m_V.nb_rows(), l);
    //         //     }
    //         // }
    //         // LowRankMatrix temp(Uconc, Vconc);
    //         /// la il faudrait une QRSVD
    //         // en attendant ACA sur M+N
    //         Matrix<CoefficientPrecision> mat(m_U.nb_rows(), m_V.nb_cols());
    //         m_U.add_matrix_product('N', 1.0, m_V.data(), 1.0, mat.data(), m_V.nb_cols());
    //         u.add_matrix_product('N', 1.0, v.data(), 1.0, mat.data(), v.nb_cols());
    //         MatrixGenerator<CoefficientPrecision> gen(mat, 0, 0);
    //         LowRankMatrix lr(gen, LRGenerator, t, s);
    //         m_U    = lr.Get_U();
    //         m_V    = lr.Get_V();
    //         m_rank = lr.rank_of();
    //     }
    // }

    LowRankMatrix actualise(const Matrix<CoefficientPrecision> &u, const Matrix<CoefficientPrecision> &v, VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision> &LRGenerator, const Cluster<CoordinatesPrecision> &t, const Cluster<CoordinatesPrecision> &s) const {
        if (u.nb_rows() == this->m_U.nb_rows() and v.nb_cols() == this->m_V.nb_cols()) {

            // Matrix<CoefficientPrecision> Uconc(u.nb_rows(), m_U.nb_cols() + u.nb_cols());
            // Matrix<CoefficientPrecision> Vconc(m_V.nb_rows() + v.nb_rows(), v.nb_cols());
            // // U= (U1, U2) , V= (V1, V2) ;
            // for (int k = 0; k < u.nb_rows(); ++k) {
            //     for (int l = 0; l < this->Get_U().nb_cols(); ++l) {
            //         Uconc(k, l) = m_U(k, l);
            //     }
            //     for (int l = m_U.nb_cols(); l < m_U.nb_cols() + u.nb_cols(); ++l) {
            //         Uconc(k, l) = u(k, l - m_U.nb_cols());
            //     }
            // }
            // for (int l = 0; l < v.nb_cols(); ++l) {
            //     for (int k = 0; k < m_V.nb_rows(); ++k) {
            //         Vconc(k, l) = m_V(k, l);
            //     }
            //     for (int k = m_V.nb_rows(); k < m_V.nb_rows() + v.nb_rows(); ++k) {
            //         Vconc(k, l) = v(k - m_V.nb_rows(), l);
            //     }
            // }
            // LowRankMatrix temp(Uconc, Vconc);
            /// la il faudrait une QRSVD
            // en attendant ACA sur M+N
            Matrix<CoefficientPrecision> mat(m_U.nb_rows(), m_V.nb_cols());
            m_U.add_matrix_product('N', 1.0, m_V.data(), 1.0, mat.data(), m_V.nb_cols());
            u.add_matrix_product('N', 1.0, v.data(), 1.0, mat.data(), v.nb_cols());
            MatrixGenerator<CoefficientPrecision> gen(mat, 0, 0);
            LowRankMatrix lr(gen, LRGenerator, t, s);
            return lr;
        } else {
            std::cerr << "wrong size" << std::endl;
        }
        return *this;
    }
    void add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const {
        if (m_rank == 0) {
            std::fill(out, out + m_U.nb_cols(), 0);
        } else if (trans == 'N') {
            std::vector<CoefficientPrecision> a(m_rank);
            m_V.add_vector_product(trans, 1, in, 0, a.data());
            m_U.add_vector_product(trans, alpha, a.data(), beta, out);
        } else {
            std::vector<CoefficientPrecision> a(m_rank);
            m_U.add_vector_product(trans, 1, in, 0, a.data());
            m_V.add_vector_product(trans, alpha, a.data(), beta, out);
        }
    }

    void add_matrix_product(char transa, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const {
        if (m_rank == 0) {
            std::fill(out, out + m_V.nb_cols() * mu, 0);
        } else if (transa == 'N') {
            std::vector<CoefficientPrecision> a(m_rank * mu);
            m_V.add_matrix_product(transa, 1, in, 0, a.data(), mu);
            m_U.add_matrix_product(transa, alpha, a.data(), beta, out, mu);
        } else {
            std::vector<CoefficientPrecision> a(m_rank * mu);
            m_U.add_matrix_product(transa, 1, in, 0, a.data(), mu);
            m_V.add_matrix_product(transa, alpha, a.data(), beta, out, mu);
        }
    }

    void add_matrix_product_row_major(char transa, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const {
        if (m_rank == 0) {
            std::fill(out, out + m_V.nb_cols() * mu, 0);
        } else if (transa == 'N') {
            std::vector<CoefficientPrecision> a(m_rank * mu);
            m_V.add_matrix_product_row_major(transa, 1, in, 0, a.data(), mu);
            m_U.add_matrix_product_row_major(transa, alpha, a.data(), beta, out, mu);
        } else {
            std::vector<CoefficientPrecision> a(m_rank * mu);
            m_U.add_matrix_product_row_major(transa, 1, in, 0, a.data(), mu);
            m_V.add_matrix_product_row_major(transa, alpha, a.data(), beta, out, mu);
        }
    }

    void
    mvprod(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
        if (m_rank == 0) {
            std::fill(out, out + m_U.nb_cols(), 0);
        } else {
            std::vector<CoefficientPrecision> a(m_rank);
            m_V.mvprod(in, a.data());
            m_U.mvprod(a.data(), out);
        }
    }

    void add_mvprod_row_major(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu, char transb = 'T', char op = 'N') const {
        if (m_rank != 0) {
            std::vector<CoefficientPrecision> a(m_rank * mu);
            if (op == 'N') {
                m_V.mvprod_row_major(in, a.data(), mu, transb, op);
                m_U.add_mvprod_row_major(a.data(), out, mu, transb, op);
            } else if (op == 'C' || op == 'T') {
                m_U.mvprod_row_major(in, a.data(), mu, transb, op);
                m_V.add_mvprod_row_major(a.data(), out, mu, transb, op);
            }
        }
    }

    void copy_to_dense(CoefficientPrecision *const out) const {
        char transa                = 'N';
        char transb                = 'N';
        int M                      = m_U.nb_rows();
        int N                      = m_V.nb_cols();
        int K                      = m_U.nb_cols();
        CoefficientPrecision alpha = 1;
        int lda                    = m_U.nb_rows();
        int ldb                    = m_V.nb_rows();
        CoefficientPrecision beta  = 0;
        int ldc                    = m_U.nb_rows();

        Blas<CoefficientPrecision>::gemm(&transa, &transb, &M, &N, &K, &alpha, m_U.data(), &lda, m_V.data(), &ldb, &beta, out, &ldc);
    }

    double compression_ratio() const {
        return (m_number_of_rows * m_number_of_columns) / (double)(m_rank * (m_number_of_rows + m_number_of_columns));
    }

    double space_saving() const {
        return (1 - (m_rank * (1. / double(m_number_of_rows) + 1. / double(m_number_of_columns))));
    }

    friend std::ostream &operator<<(std::ostream &os, const LowRankMatrix &m) {
        os << "rank:\t" << m.rank << std::endl;
        os << "number_of_rows:\t" << m.m_number_of_rows << std::endl;
        os << "m_number_of_columns:\t" << m.m_number_of_columns << std::endl;
        os << "U:\n";
        os << m.m_U << std::endl;
        os << m.m_V << std::endl;

        return os;
    }

    LowRankMatrix lrmult(const LowRankMatrix *B) {
        auto U = m_U;
        auto V = m_V * (B->Get_U()) * (B->Get_V());
        return LowRankMatrix(U, V);
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
underlying_type<CoefficientPrecision> Frobenius_relative_error(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, const LowRankMatrix<CoefficientPrecision, CoordinatePrecision> &lrmat, const VirtualGenerator<CoefficientPrecision> &ref, int reqrank = -1) {
    assert(reqrank <= lrmat.rank_of());
    if (reqrank == -1) {
        reqrank = lrmat.rank_of();
    }
    underlying_type<CoefficientPrecision> norm = 0;
    underlying_type<CoefficientPrecision> err  = 0;
    std::vector<CoefficientPrecision> aux(lrmat.nb_rows() * lrmat.nb_cols());
    ref.copy_submatrix(target_cluster, source_cluster, aux.data());
    for (int j = 0; j < lrmat.nb_rows(); j++) {
        for (int k = 0; k < lrmat.nb_cols(); k++) {
            norm += std::pow(std::abs(aux[j + k * lrmat.nb_rows()]), 2);
            for (int l = 0; l < reqrank; l++) {
                aux[j + k * lrmat.nb_rows()] = aux[j + k * lrmat.nb_rows()] - lrmat.get_U(j, l) * lrmat.get_V(l, k);
            }
            err += std::pow(std::abs(aux[j + k * lrmat.nb_rows()]), 2);
        }
    }
    err = err / norm;
    return std::sqrt(err);
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
underlying_type<CoefficientPrecision> Frobenius_absolute_error(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, const LowRankMatrix<CoefficientPrecision, CoordinatePrecision> &lrmat, const VirtualGenerator<CoefficientPrecision> &ref, int reqrank = -1) {
    assert(reqrank <= lrmat.rank_of());
    if (reqrank == -1) {
        reqrank = lrmat.rank_of();
    }
    underlying_type<CoefficientPrecision> err = 0;
    Matrix<CoefficientPrecision> aux(lrmat.nb_rows(), lrmat.nb_cols());
    ref.copy_submatrix(target_cluster.get_size(), source_cluster.get_size(), target_cluster.get_offset(), source_cluster.get_offset(), aux.data());

    for (int j = 0; j < lrmat.nb_rows(); j++) {
        for (int k = 0; k < lrmat.nb_cols(); k++) {

            for (int l = 0; l < reqrank; l++) {
                aux(j, k) = aux(j, k) - lrmat.get_U(j, l) * lrmat.get_V(l, k);
            }
            err += std::pow(std::abs(aux(j, k)), 2);
        }
    }
    return std::sqrt(err);
}

} // namespace htool

#endif
