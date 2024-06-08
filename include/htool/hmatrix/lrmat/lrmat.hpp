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
    /// je rajoute l'addition tronqué , ca renvoie U, V avec eventuellement V=0 si il y a pas d'approx de rang faible diponible
    /// flage = true , return false si ca a pas marché ou que c'était pas interressant ( c'est le cas extreme (10, 5)*(5, 10) +(10, 5)*(5, 10))
    ///////////////////
    LowRankMatrix formatted_addition(const LowRankMatrix &R, const double &epsilon0, bool &flag) {
        int conc_nc  = this->rank_of() + R.rank_of();
        int interest = (m_number_of_rows + m_number_of_columns) / 2.0;
        if (conc_nc > interest) {
            flag = false;
            return *this;
        } else {
            auto U1 = this->Get_U();
            auto V1 = this->Get_V();
            auto U2 = R.Get_U();
            auto V2 = R.Get_V();

            auto Uconc = conc_col(U1, U2);
            auto Vconc = conc_row_T(V1, V2);
            auto QRu   = QR_factorisation(U1.nb_rows(), Uconc.nb_cols(), Uconc);
            auto QRv   = QR_factorisation(Vconc.nb_rows(), Vconc.nb_cols(), Vconc);
            Matrix<double> RuRv(QRu[0].nb_rows(), QRv[0].nb_cols());
            int ldu = QRu[1].nb_rows();
            int ldv = QRv[1].nb_cols();

            int rk       = QRu[1].nb_cols();
            double alpha = 1.0;
            double beta  = 1.0;
            int ldc      = std::max(ldu, ldv);
            Blas<double>::gemm("N", "T", &ldu, &ldv, &rk, &alpha, QRu[1].data(), &ldu, QRv[1].data(), &ldv, &beta, RuRv.data(), &ldc);
            Matrix<double> svdU(RuRv.nb_rows(), std::min(RuRv.nb_rows(), RuRv.nb_cols()));
            Matrix<double> svdV(std::min(RuRv.nb_rows(), RuRv.nb_cols()), RuRv.nb_cols());
            auto S        = compute_svd(RuRv, svdU, svdV);
            double margin = norm2(S);
            auto it       = std::find_if(S.begin(), S.end(), [epsilon0, margin](double s) {
                return s < (epsilon0 * margin);
            });
            Matrix<double> Ss(S.size(), S.size());
            for (int k = 0; k < S.size(); ++k) {
                Ss(k, k) = S[k];
            }
            int rep = std::distance(S.begin(), it);
            // int rep = S.size();
            // std::cout << " erreur spectrale de : " << S[rep] << ", normalized : " << S[rep + 1] / margin << "//" << rep << ',' << S.size() << std::endl;
            if (rep < conc_nc) {
                Matrix<double> Urestr(svdU.nb_rows(), rep);
                for (int l = 0; l < rep; ++l) {
                    Urestr.set_col(l, svdU.get_col(l));
                }
                Matrix<double> Vrestr(rep, svdV.nb_cols());
                for (int k = 0; k < rep; ++k) {
                    Vrestr.set_row(k, svdV.get_row(k));
                }
                Matrix<double> srestr(rep, rep);
                for (int k = 0; k < rep; ++k) {
                    srestr(k, k) = S[k];
                }
                // std::cout << "erreur svd :  " << normFrob((svdU * Ss * svdV)) << "    " << normFrob(Urestr * srestr * Vrestr) << std::endl;
                auto res_U = QRu[0] * Urestr * srestr;
                auto res_V = Vrestr * Vrestr.transp(QRv[0]);
                LowRankMatrix res(res_U, res_V);
                return res;
            } else {
                flag = false;
                return *this;
            }
        }
    }

    LowRankMatrix
    actualise(const Matrix<CoefficientPrecision> &u, const Matrix<CoefficientPrecision> &v, VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision> &LRGenerator, const Cluster<CoordinatesPrecision> &t, const Cluster<CoordinatesPrecision> &s) const {
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

    void mvprod(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
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

    //////////////////////////////////////
    //// Somme de deux LR mat
    std::vector<Matrix<CoefficientPrecision>> compute_lr_update(const Matrix<CoefficientPrecision> &A, const Matrix<CoefficientPrecision> &B, const int &rk, const CoefficientPrecision &epsilon) {
        int target_size = A.nb_rows();
        int source_size = B.nb_cols();
        std::vector<Matrix<CoefficientPrecision>> res;
        auto &U     = this->Get_U();
        auto &V     = this->Get_V();
        const int N = U.nb_cols() + A.nb_cols(); // rank1+rank2
        // on regarde si c'est interressant de concaténer
        if (U.nb_rows() > N && V.nb_cols() > N) {
            // on concatène
            auto big_VT = conc_row_T(V, B);
            auto big_U  = conc_col(U, A);
            std::cout << "size : bigu  " << big_U.nb_rows() << ',' << big_U.nb_cols() << std::endl;
            std::cout << "size big_V " << big_VT.nb_rows() << ',' << big_VT.nb_cols() << std::endl;
            std::cout << " n " << N << std::endl;
            // QR sur big_U
            std::vector<Matrix<CoefficientPrecision>> QRu;
            auto tempu = big_U;
            QRu        = QR_factorisation(big_U.nb_rows(), big_U.nb_cols(), big_U);
            auto Qu    = QRu[0];
            auto Ru    = QRu[1];
            // std::cout << "erreur QR sur U " << normFrob(Qu * Ru - tempu) / normFrob(tempu) << std::endl;
            // QR sur big_V transposé ----------> WARNING : on est sur big_V^T pas big_V, il faut retranposer après
            //->on doit transposé
            auto QRv   = QR_factorisation(big_VT.nb_rows(), big_VT.nb_cols(), big_VT);
            auto tempv = big_VT;
            auto Qv    = QRv[0];
            auto Rv    = QRv[1];
            // std::cout << "erreur QR sur V " << normFrob(Qv * Rv - big_VT) / normFrob(big_VT) << std::endl;
            // Ru*Rv
            Matrix<CoefficientPrecision> R_uR_v(N, N);
            double alpha_0 = 1.0;
            double beta_0  = 0.0;
            R_uR_v         = Ru * Rv.transp(Rv);
            // Blas<CoefficientPrecision>::gemm("N", "T", &N, &N, &N, &alpha_0, Ru.data(), &N, Rv.data(), &N, &beta_0, R_uR_v.data(), &N); // j'ai besoin du "transb" du coup je prend gemm
            // truncated SVD sur R_uR_v
            Matrix<CoefficientPrecision> Uu(N, N);
            Matrix<CoefficientPrecision> VT(N, N);
            auto S = compute_svd(R_uR_v, Uu, VT);
            // truncation index
            int trunc = 0;
            while (S[trunc] > epsilon) {
                trunc += 1;
            }
            // on regarde si il y a un rang d'approximation potable (j'ai pris N/2 c'est arbitraire)
            // ca peut arriver en sommant deux petites lr que le rang soit plus faible
            if (trunc < N / 2) {
                Matrix<CoefficientPrecision> Strunc(trunc, trunc);
                for (int k = 0; k < trunc; ++k) {
                    Strunc(k, k) = S[k];
                }
                ///// truncation of svd vectors
                auto Utrunc = Uu.trunc_col(trunc);
                auto Vtrunc = VT.trunc_row(trunc);
                ///// fast trunc
                // Matrix<CoefficientPrecision> res_V(trunc, source_size);
                // Blas<CoefficientPrecision>::gemm("N", "T", &trunc, &N, &source_size, &alpha_0, Vtrunc.data(), &N, Qv.data(), &source_size, &beta_0, res_V.data(), &std::max(trunc, source_size)); // j'ai besoin du "transb" du coup je prend gemm
                auto res_V = Vtrunc * Qv.transp(Qv);
                // Blas<CoefficientPrecision>::gemm("N", "T", &trunc, &source_size, &trunc, &alpha_0, Vtrunc.data(), &trunc, big_VT.data(), &source_size, &beta_0, res_V.data(), &source_size);
                // auto res_V = Vtrunc * Qv.transp(Qv);
                // auto res_U = Qu * Utrunc * Strunc;
                Matrix<CoefficientPrecision> Us(N, trunc);
                // Blas<CoefficientPrecision>::gemm("N", "N", &N, &trunc, &trunc, &alpha_0, Utrunc.data(), &N, Strunc.data(), &trunc, &beta_0, Us.data(), &N);
                Matrix<CoefficientPrecision> res_U(target_size, trunc);
                // Blas<CoefficientPrecision>::gemm("N", "N", &target_size, &trunc, &N, &alpha_0, Qu.data(), &target_size, Us.data(), &N, &beta_0, Us.data(), &target_size);
                res_U = Qu * Utrunc * Strunc;
                // Finally
                res.push_back(res_U);
                res.push_back(res_V);
                std::cout << "on push U et V de taille " << res_U.nb_rows() << ',' << res_U.nb_cols() << ',' << res_V.nb_rows() << ',' << res_V.nb_cols() << std::endl;
            }
        } else {
            // concaténation pas interressante
            // on doit differencier nr > nc ..........

            auto big_mat = U * V + A * B;
            if (big_mat.nb_rows() >= big_mat.nb_cols()) {
                int ld = std::max(big_mat.nb_rows(), big_mat.nb_cols());
                int Nn = std::min(big_mat.nb_rows(), big_mat.nb_cols());
                std::vector<CoefficientPrecision> S(Nn, 0.0);
                Matrix<CoefficientPrecision> Uu(big_mat.nb_rows(), big_mat.nb_rows());
                Matrix<CoefficientPrecision> VT(big_mat.nb_cols(), big_mat.nb_cols());
                int info;
                int lwork = 10 * ld;
                std::vector<double> rwork(lwork);
                std::vector<CoefficientPrecision> work(lwork);
                int nr = big_mat.nb_rows();
                int nc = big_mat.nb_cols();
                Lapack<CoefficientPrecision>::gesvd("A", "A", &nr, &nc, big_mat.data(), &ld, S.data(), Uu.data(), &nr, VT.data(), &nc, work.data(), &lwork, rwork.data(), &info);
                int trunc      = 0;
                double alpha_0 = 1.0;
                double beta_0  = 0.0;
                while (S[trunc] > epsilon) {
                    trunc += 1;
                }
                if (trunc < Nn / 2) {
                    Matrix<CoefficientPrecision> Strunc(trunc, trunc);
                    for (int k = 0; k < trunc; ++k) {
                        Strunc(k, k) = S[k];
                    }
                    ///// truncation of svd vectors
                    auto Utrunc = Uu.trunc_col(trunc);
                    auto Vtrunc = VT.trunc_row(trunc);

                    Matrix<CoefficientPrecision> Us(nr, trunc);
                    // Blas<CoefficientPrecision>::gemm("N", "N", &nr, &nc, &trunc, &alpha_0, Utrunc.data(), &nr, Strunc.data(), &trunc, &beta_0, Us.data(), &std::max(nr, nc)); // j'ai besoin du "transb" du coup je prend gemm
                    Matrix<CoefficientPrecision> Ss(Us.nb_cols(), Us.nb_cols());
                    for (int k = 0; k < Us.nb_cols(); ++k) {
                        Ss(k, k) = S[k];
                    }
                    Us = Utrunc * Ss;
                    res.push_back(Us);
                    res.push_back(Vtrunc);
                    std::cout << "on push U et V de taille " << Us.nb_rows() << ',' << Us.nb_cols() << ',' << Vtrunc.nb_rows() << ',' << Vtrunc.nb_cols() << std::endl;

                } else {
                    // std::cout << "pas de bonne approximation de UV +AB" << std::endl;
                }
            } else {
                std::cout << "nc> nr" << std::endl;
            }
        }
        return res;
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

template <typename CoefficientPrecision>
std::vector<Matrix<CoefficientPrecision>> formatted_soustraction(const Matrix<CoefficientPrecision> &U1, const Matrix<CoefficientPrecision> &V1, const Matrix<CoefficientPrecision> &U2, const Matrix<CoefficientPrecision> &V2, const double &epsilon, const int &offset_t, const int &offset_s) {
    Matrix<CoefficientPrecision> res_U, res_V;
    std::vector<Matrix<CoefficientPrecision>> res;
    /// onr egarde si ca vaut le coup de concatener
    double alpha = 1.0;
    double beta  = 0.0;

    int conc_nc = U1.nb_cols() + U2.nb_cols();
    if (conc_nc < (U1.nb_rows() + V1.nb_cols()) / 2) {
        // on concatène
        auto refu  = -1.0 * U2;
        auto Uconc = conc_col(U1, refu);
        auto Vconc = conc_row_T(V1, V2);
        // QR sur U1, U2 et sur (V1, V2)^T ( nr doit être > nc donc on doit faire QR de la transposé pour Vconc )
        auto QRu = QR_factorisation(U1.nb_rows(), Uconc.nb_cols(), Uconc);
        auto QRv = QR_factorisation(Vconc.nb_rows(), Vconc.nb_cols(), Vconc);
        Matrix<CoefficientPrecision> RuRv(QRu[1].nb_rows(), QRv[1].nb_rows());
        // Ru*Rv^T
        int nr = QRu[1].nb_rows();
        int nc = QRu[1].nb_cols();
        int nl = RuRv.nb_cols();
        Blas<CoefficientPrecision>::gemm("N", "T", &nr, &nl, &nc, &alpha, QRu[1].data(), &nr, QRv[1].data(), &nc, &alpha, RuRv.data(), &nl);
        // std::cout << "erreur RURV^T" << normFrob(RuRv - QRu[1] * QRv[1].transp(QRv[1])) / normFrob(RuRv) << std::endl;
        Matrix<CoefficientPrecision> svdU(RuRv.nb_rows(), std::min(RuRv.nb_rows(), RuRv.nb_cols()));
        Matrix<CoefficientPrecision> svdV(std::min(RuRv.nb_rows(), RuRv.nb_cols()), RuRv.nb_cols());
        // svd sur RuRv
        // auto temppp = RuRv;
        auto S    = compute_svd(RuRv, svdU, svdV);
        int rep   = 0;
        bool flag = true;
        std::vector<CoefficientPrecision> Srestr;
        // k premières plus grandes que epsilon
        while (flag) {
            for (int k = 0; k < S.size(); ++k) {
                Srestr.push_back(S[k]);
                if (S[k] < epsilon) {
                    rep  = k;
                    flag = false;
                }
            }
            // rep += 1;
            // std::cout << "onr s'est arreté sur " << S[rep] << "  , aprés : " << S[rep + 1] << std::endl;
            flag = false;
        }
        // Matrix<CoefficientPrecision> SS(S.size(), S.size());
        // for (int k = 0; k < S.size(); ++k) {
        //     SS(k, k) = S[k];
        // }
        // std::cout << "erreur SVD " << normFrob(svdU * SS * svdV - temppp) / normFrob(temppp) << std::endl;
        // on regarde si on a trouvé un petit rang
        if (rep < (U1.nb_rows() + V1.nb_cols()) / 2) {
            auto Urestr = svdU.trunc_col(rep + 1);
            auto Vrestr = svdV.trunc_row(rep + 1);
            Matrix<CoefficientPrecision> srestr(rep + 1, rep + 1);
            for (int k = 0; k < rep + 1; ++k) {
                srestr(k, k) = Srestr[k];
            }
            // std::cout << "erreur svd tronqué : " << normFrob(Urestr * srestr * Vrestr - temppp) / normFrob(temppp) << std::endl;
            Matrix<CoefficientPrecision> temp(Urestr.nb_cols(), rep + 1);
            res_U.resize(QRu[0].nb_rows(), rep + 1);
            Urestr.add_matrix_product('N', alpha, srestr.data(), beta, temp.data(), rep + 1);
            QRu[0].add_matrix_product('N', alpha, temp.data(), beta, res_U.data(), rep + 1);
            // Blas<CoefficientPrecision>::gemm('N', 'N', Urestr.nb_rows(), rep + 1, rep + 1, 1.0, Urestr.data(), Urestr.nb_rows(), srestr.data(), rep + 1, 1.0, temp.data(), Urestr.nb_rows());
            // Blas<CoefficientPrecision>::gemm('N', 'N', QRu[0].nb_rows(), rep + 1, Urestr.nb_rows(), 1.0, QRu[0].data(), std::max(QRu[0].nb_rows(), QRu[0].nb_cols()), temp.data(), Urestr.nb_rows(), 1.0, res_U.data(), res_U.nb_rows());
            res_V.resize(Vrestr.nb_rows(), QRv[0].nb_rows());
            int nr    = Vrestr.nb_rows();
            int nc    = Vrestr.nb_cols();
            int nl    = QRv[0].nb_rows();
            auto test = Vrestr * QRv[0].transp(QRv[0]);

            Blas<CoefficientPrecision>::gemm("N", "T", &nr, &nl, &nc, &alpha, Vrestr.data(), &nr, QRv[0].data(), &nl, &alpha, res_V.data(), &nr);
            // std::cout << " erreur VQ^T = " << normFrob(test - res_V) << std::endl;
            // std::cout << " 0   on push U et V de norme : " << normFrob(res_U) << ',' << normFrob(res_V) << std::endl;
            res.push_back(res_U);
            res.push_back(res_V);
        } else {
            // il y a pas de petit rang , c'est pas interresant de faire les restrictions, tout ca servait a rien on fait juste U1V1+U2V2
            double malpha = -1;
            res_U.resize(U1.nb_rows(), V1.nb_cols());
            U2.add_matrix_product('N', alpha, V2.data(), alpha, res_U.data(), V2.nb_cols());
            U1.add_matrix_product('N', alpha, V1.data(), malpha, res_U.data(), V1.nb_cols());

            // Blas<CoefficientPrecision>::gemm('N', 'N', U1.nb_rows(), V1.nb_cols(), U1.nb_cols(), 1.0, U1.data(), U1.nb_rows(), V1.data(), V1.nb_cols(), 1.0, res_U.data(), std::max(U1.nb_rows(), V1.nb_cols()));
            // Blas<CoefficientPrecision>::gemm('N', 'N', U2.nb_rows(), V2.nb_cols(), U2.nb_cols(), 1.0, U2.data(), U2.nb_rows(), V2.data(), V2.nb_cols(), 1.0, res_U.data(), std::max(U2.nb_rows(), V2.nb_cols()));
            res.push_back(res_U);
        }
    } else {
        // ca sert a rien de concaténer on fait + dense et on regarde si il y a quand même pas une approx
        Matrix<CoefficientPrecision> temp(U1.nb_rows(), V1.nb_cols());
        double malpha = -1;

        U2.add_matrix_product('N', alpha, V2.data(), alpha, temp.data(), V2.nb_cols());

        U1.add_matrix_product('N', alpha, V1.data(), malpha, temp.data(), V1.nb_cols());
        // Lapack<CoefficientPrecision>::gemm('N', 'N', U1.nb_rows(), V1.nb_cols(), U1.nb_cols(), 1.0, U1.data(), U1.nb_rows(), V1.data(), V1.nb_cols(), 1.0, temp.data(), std::max(temp.nb_rows(), temp.nb_cols()));
        // Lapack<CoefficientPrecision>::gemm('N', 'N', U2.nb_rows(), V2.nb_cols(), U2.nb_cols(), 1.0, U2.data(), U2.nb_rows(), V2.data(), V2.nb_cols(), 1.0, temp.data(), std::max(temp.nb_rows(), temp.nb_cols()));
        Matrix<CoefficientPrecision> uu, vv;
        ACA(temp, offset_t, offset_s, epsilon, -1, uu, vv);
        if (uu.nb_cols() == 0) {
            res.push_back(temp);
        } else {
            // std::cout << "  1     on push U et V de norme : " << normFrob(uu) << ',' << normFrob(vv) << std::endl;
            res.push_back(uu);
            res.push_back(vv);
        }
    }
    return res;
}

} // namespace htool

#endif
