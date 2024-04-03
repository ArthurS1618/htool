#ifndef SUM_EXPRESSIONS_HPP
#define SUM_EXPRESSIONS_HPP
#include "../basic_types/vector.hpp"
#include <random>

namespace htool {

template <class CoefficientPrecision>
int get_pivot(const std::vector<CoefficientPrecision> &x, const int &previous_pivot) {
    double delta = 0.0;
    int pivot    = 0;
    for (int k = 0; k < x.size(); ++k) {
        if (std::abs(x[k]) > delta) {
            // std::cout << "???????????????,,," << std::endl;

            if (k != previous_pivot) {
                pivot = k;
                delta = std::abs(x[k]);
            }
        }
    }
    if (delta == 0.0) {
        return -1;
    } else {
        // std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        return pivot;
    }
}

std::vector<double> generate_random_vector(int size) {
    std::random_device rd;  // Source d'entropie aléatoire
    std::mt19937 gen(rd()); // Générateur de nombres pseudo-aléatoires

    std::uniform_real_distribution<double> dis(1.0, 10.0); // Plage de valeurs pour les nombres aléatoires (ici de 1.0 à 10.0)

    std::vector<double> random_vector;
    random_vector.reserve(size); // Allocation de mémoire pour le vecteur

    for (int i = 0; i < size; ++i) {
        random_vector.push_back(dis(gen)); // Ajout d'un nombre aléatoire dans la plage à chaque itération
    }

    return random_vector;
}
template <class CoefficientPrecision>
class DenseGenerator : public VirtualGenerator<CoefficientPrecision> {
  private:
    const Matrix<CoefficientPrecision> &mat;

  public:
    DenseGenerator(const Matrix<CoefficientPrecision> &mat0) : mat(mat0) {}

    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = mat(i + row_offset, j + col_offset);
            }
        }
    }
};

template <class CoordinatePrecision>
class Cluster;
template <typename CoefficientPrecision, typename CoordinatePrecision>
class HMatrix;
template <typename CoefficientPrecision, typename CoordinatePrecision>
class LowRankMatrix;
template <typename CoefficientPrecision, typename CoordinatePrecision>
class SumExpression : public VirtualGenerator<CoefficientPrecision> {
  private:
    using HMatrixType = HMatrix<CoefficientPrecision, CoordinatePrecision>;
    using LRtype      = LowRankMatrix<CoefficientPrecision, CoordinatePrecision>;

    std::vector<Matrix<CoefficientPrecision>> Sr;
    std::vector<int> offset;     // target de u et source de v -> 2*size Sr
    std::vector<int> mid_offset; // le rho au ca ou on essaye de cut des blocs dense
    std::vector<const HMatrixType *> Sh;
    int target_size;
    int target_offset;
    int source_size;
    int source_offset;
    bool restrictible = true;

  public:
    SumExpression(const HMatrixType *A, const HMatrixType *B) {
        Sh.push_back(A);
        Sh.push_back(B);
        target_size   = A->get_target_cluster().get_size();
        target_offset = A->get_target_cluster().get_offset();
        source_size   = B->get_source_cluster().get_size();
        source_offset = B->get_source_cluster().get_offset();
    }

    SumExpression(const std::vector<Matrix<CoefficientPrecision>> &sr, const std::vector<const HMatrixType *> &sh, const std::vector<int> &offset0, const int &target_size0, const int &target_offset0, const int &source_size0, const int &source_offset0, const bool flag) {
        Sr            = sr;
        Sh            = sh;
        offset        = offset0;
        target_size   = target_size0;
        target_offset = target_offset0;
        source_size   = source_size0;
        source_offset = source_offset0;
        restrictible  = flag;
    }

    SumExpression(const std::vector<Matrix<CoefficientPrecision>> &sr, const std::vector<const HMatrixType *> &sh, const std::vector<int> &offset0, const std::vector<int> &mid_offset0, const int &target_size0, const int &target_offset0, const int &source_size0, const int &source_offset0, bool flag) {
        Sr            = sr;
        Sh            = sh;
        offset        = offset0;
        target_size   = target_size0;
        target_offset = target_offset0;
        source_size   = source_size0;
        source_offset = source_offset0;
        mid_offset    = mid_offset0;
        restrictible  = flag;
    }

    const std::vector<Matrix<CoefficientPrecision>> get_sr() const { return Sr; }
    const std::vector<const HMatrixType *> get_sh() const { return Sh; }
    int get_nr() const { return target_size; }
    int get_nc() const { return source_size; }
    int get_target_offset() const { return target_offset; }
    int get_source_offset() const { return source_offset; }
    int get_target_size() const { return target_size; }
    int get_source_size() const { return source_size; }
    const std::vector<int> get_offset() const { return offset; }
    void set_sr(const std::vector<Matrix<CoefficientPrecision>> &sr) {
        Sr = sr;
    }
    void add_HK(const HMatrixType *H, const HMatrixType *K) {
        Sh.push_back(H);
        Sh.push_back(K);
    }

    const std::vector<CoefficientPrecision> prod(const std::vector<CoefficientPrecision> &x) const {
        std::vector<CoefficientPrecision> y(target_size, 0.0);
        for (int k = 0; k < Sr.size() / 2; ++k) {
            const Matrix<CoefficientPrecision> U = Sr[2 * k];
            const Matrix<CoefficientPrecision> V = Sr[2 * k + 1];
            int oft                              = offset[2 * k];
            int ofs                              = offset[2 * k + 1];
            Matrix<CoefficientPrecision> urestr;
            Matrix<CoefficientPrecision> vrestr;
            CoefficientPrecision *ptru = new CoefficientPrecision[target_size, U.nb_cols()];
            CoefficientPrecision *ptrv = new CoefficientPrecision[V.nb_rows(), source_size];
            /// il y en a un des deux ou on devrait pouvoir faire un v.start = n*offset
            for (int i = 0; i < target_size; ++i) {
                for (int j = 0; j < U.nb_cols(); ++j) {
                    ptru[i + target_size * j] = U(i + target_offset - oft, j);
                }
            }
            for (int i = 0; i < V.nb_rows(); ++i) {
                for (int j = 0; j < source_size; ++j) {
                    ptrv[i + V.nb_rows() * j] = V(i, j + source_offset - ofs);
                }
            }
            urestr.assign(target_size, U.nb_cols(), ptru, true);
            vrestr.assign(V.nb_rows(), source_size, ptrv, true);
            // y = y + urestr * vrestr * x;
            std::vector<CoefficientPrecision> ytemp(vrestr.nb_cols(), 0.0);
            vrestr.add_vector_product('N', 1.0, x.data(), 0.0, ytemp.data());
            urestr.add_vector_product('N', 1.0, ytemp.data(), 1.0, y.data());
        }
        // std::cout << "sr ok" << std::endl;
        for (int k = 0; k < Sh.size() / 2; ++k) {
            const HMatrixType &H = *Sh[2 * k];
            const HMatrixType &K = *Sh[2 * k + 1];
            // Matrix<CoefficientPrecision> H_dense(H.get_target_cluster().get_size(), H.get_source_cluster().get_size());
            // Matrix<CoefficientPrecision> K_dense(K.get_target_cluster().get_size(), K.get_source_cluster().get_size());
            // copy_to_dense(H, H_dense.data());
            // copy_to_dense(K, K_dense.data());
            // y = y + H_dense * (K_dense * x);
            std::vector<CoefficientPrecision> y_temp(K.get_target_cluster().get_size(), 0.0);
            K.add_vector_product('N', 1.0, x.data(), 0.0, y_temp.data());
            H.add_vector_product('N', 1.0, y_temp.data(), 1.0, y.data());
        }
        return (y);
    }

    /////////////////////////////////////////////////////////////////////////
    /// NEW PROD
    /// un peu plus rapide que prd
    ////////////////////////////////////////////////////////////
    std::vector<CoefficientPrecision> dumb_prod(const char T, const std::vector<CoefficientPrecision> &x) const {
        std::vector<CoefficientPrecision> y;
        if (T == 'N') {
            Matrix<CoefficientPrecision> dense(target_size, source_size);
            this->copy_submatrix(target_size, source_size, 0, 0, dense.data());
            y.resize(target_size);
            dense.add_vector_product('N', 1.0, x.data(), 1.0, y.data());
            return y;
        } else {
            Matrix<CoefficientPrecision> dense(target_size, source_size);
            this->copy_submatrix(target_size, source_size, 0, 0, dense.data());
            y.resize(source_size);
            dense.add_vector_product('T', 1.0, x.data(), 1.0, y.data());
        }
        return y;
    }

    // prod pour sum expr avec accumulate update
    // std::vector<CoefficientPrecision> new_prod(const char T, const std::vector<CoefficientPrecision> &x) const {
    //     if (T == 'N') {
    //         std::vector<CoefficientPrecision> y(target_size, 0.0);
    //         if (Sr.size() > 0) {
    //             auto u = Sr[0];
    //             auto v = Sr[1];
    //             std::vector<CoefficientPrecision> ytemp(v.nb_rows(), 0.0);
    //             v.add_vector_product('N', 1.0, x.data(), 0.0, ytemp.data());
    //             u.add_vector_product('N', 1.0, ytemp.data(), 1.0, y.data());
    //         }

    //         for (int k = 0; k < Sh.size() / 2; ++k) {
    //             auto H = Sh[2 * k];
    //             auto K = Sh[2 * k + 1];
    //             std::vector<CoefficientPrecision> y_temp(K->get_target_cluster().get_size(), 0.0);
    //             K->add_vector_product('N', 1.0, x.data(), 0.0, y_temp.data());
    //             H->add_vector_product('N', 1.0, y_temp.data(), 1.0, y.data());
    //         }
    //         return (y);
    //     }

    //     else {
    //         std::vector<CoefficientPrecision> y(source_size, 0.0);
    //         if (Sr.size() > 0) {
    //             auto u = Sr[0];
    //             auto v = Sr[1];
    //             std::vector<CoefficientPrecision> ytemp(u.nb_cols(), 0.0);
    //             u.add_vector_product('T', 1.0, x.data(), 0.0, ytemp.data());
    //             v.add_vector_product('T', 1.0, ytemp.data(), 1.0, y.data());
    //         }

    //         // y = y +(x* urestr) * vrestrx;

    //         for (int k = 0; k < Sh.size() / 2; ++k) {
    //             auto H = Sh[2 * k];
    //             auto K = Sh[2 * k + 1];
    //             std::vector<CoefficientPrecision> y_temp(H->get_source_cluster().get_size(), 0.0);
    //             H->add_vector_product('T', 1.0, x.data(), 1.0, y_temp.data());
    //             K->add_vector_product('T', 1.0, y_temp.data(), 1.0, y.data());
    //         }
    //         return (y);
    //     }
    // }

    // lui il marche mais c'est avec les listes de UV et sans accumulate update
    std::vector<CoefficientPrecision> new_prod(const char T, const std::vector<CoefficientPrecision> &x) const {
        if (T == 'N') {
            std::vector<CoefficientPrecision> y(target_size, 0.0);
            for (int k = 0; k < Sr.size() / 2; ++k) {
                const Matrix<CoefficientPrecision> U = Sr[2 * k];
                const Matrix<CoefficientPrecision> V = Sr[2 * k + 1];
                int oft                              = offset[2 * k];
                int ofs                              = offset[2 * k + 1];
                Matrix<CoefficientPrecision> urestr(target_size, U.nb_cols());
                for (int i = 0; i < target_size; ++i) {
                    for (int j = 0; j < U.nb_cols(); ++j) {
                        urestr(i, j) = U(i + target_offset - oft, j);
                    }
                }
                Matrix<CoefficientPrecision> vrestr(V.nb_rows(), source_size);
                for (int i = 0; i < V.nb_rows(); ++i) {
                    for (int j = 0; j < source_size; ++j) {
                        vrestr(i, j) = V(i, j + source_offset - ofs);
                    }
                }
                std::vector<CoefficientPrecision> ytemp(vrestr.nb_rows(), 0.0);
                vrestr.add_vector_product('N', 1.0, x.data(), 0.0, ytemp.data());
                urestr.add_vector_product('N', 1.0, ytemp.data(), 1.0, y.data());
                // y = y + urestr * (vrestr * x);
            }
            for (int k = 0; k < Sh.size() / 2; ++k) {
                auto H = Sh[2 * k];
                auto K = Sh[2 * k + 1];
                std::vector<CoefficientPrecision> y_temp(K->get_target_cluster().get_size(), 0.0);
                K->add_vector_product('N', 1.0, x.data(), 0.0, y_temp.data());
                H->add_vector_product('N', 1.0, y_temp.data(), 1.0, y.data());
            }
            return (y);
        }

        else {
            std::vector<CoefficientPrecision> y(source_size, 0.0);
            for (int k = 0; k < Sr.size() / 2; ++k) {
                const Matrix<CoefficientPrecision> U = Sr[2 * k];
                const Matrix<CoefficientPrecision> V = Sr[2 * k + 1];
                int oft                              = offset[2 * k];
                int ofs                              = offset[2 * k + 1];
                Matrix<CoefficientPrecision> urestr(target_size, U.nb_cols());
                for (int i = 0; i < target_size; ++i) {
                    for (int j = 0; j < U.nb_cols(); ++j) {
                        urestr(i, j) = U(i + target_offset - oft, j);
                    }
                }
                Matrix<CoefficientPrecision> vrestr(V.nb_rows(), source_size);
                for (int i = 0; i < V.nb_rows(); ++i) {
                    for (int j = 0; j < source_size; ++j) {
                        vrestr(i, j) = V(i, j + source_offset - ofs);
                    }
                }
                std::vector<CoefficientPrecision> ytemp(urestr.nb_cols(), 0.0);
                urestr.add_vector_product('T', 1.0, x.data(), 0.0, ytemp.data());
                vrestr.add_vector_product('T', 1.0, ytemp.data(), 1.0, y.data());

                // y = y +(x* urestr) * vrestrx;
            }
            for (int k = 0; k < Sh.size() / 2; ++k) {
                auto H = Sh[2 * k];
                auto K = Sh[2 * k + 1];
                std::vector<CoefficientPrecision> y_temp(H->get_source_cluster().get_size(), 0.0);
                H->add_vector_product('T', 1.0, x.data(), 1.0, y_temp.data());
                K->add_vector_product('T', 1.0, y_temp.data(), 1.0, y.data());
            }
            return (y);
        }
    }
    /// ///////////////////////////////////////////////
    /////// get_coeff pour copy sub_matrix
    /////////////////////////////////////////
    const CoefficientPrecision
    get_coeff(const int &i, const int &j) const {
        std::vector<CoefficientPrecision> xj(source_size, 0.0);
        xj[j]                               = 1.0;
        std::vector<CoefficientPrecision> y = this->new_prod(xj);

        return y[i];
    }

    ///////////////////////////////////////////////////////////////////
    /// Copy sub matrix ----------------> sumexpr = virtual_generator
    ////////////////////////////////////////////////////
    std::vector<CoefficientPrecision> get_col(const int &l) const {
        std::vector<double> e_l(source_size);
        e_l[l]   = 1.0;
        auto col = this->new_prod('N', e_l);
        // Matrix<CoefficientPrecision> dense(target_size, source_size);
        // this->copy_submatrix(target_size, source_size, 0, 0, dense.data());
        // auto col = dense * e_l;
        return col;
    }
    std::vector<CoefficientPrecision> get_row(const int &k) const {
        std::vector<double> e_k(target_size);
        e_k[k]   = 1.0;
        auto row = this->new_prod('T', e_k);
        // Matrix<CoefficientPrecision> dense(target_size, source_size);
        // this->copy_submatrix(target_size, source_size, 0, 0, dense.data());
        // auto row = dense.transp(dense) * e_k;
        return row;
    }

    // do Sr = Sr +AB with QR svd
    void actualise(const Matrix<CoefficientPrecision> &A, const Matrix<CoefficientPrecision> &B, const int &rk = -1, const CoefficientPrecision &epsilon = -1) {
        // first low ranks
        if (Sr.size() == 0) {
            Sr.push_back(A);
            Sr.push_back(B);
        }
        // update
        else {
            auto U = Sr[0];
            auto V = Sr[1];
            // on concatène
            Matrix<CoefficientPrecision> big_U(U.nb_rows(), U.nb_cols() + A.nb_cols());
            Matrix<CoefficientPrecision> big_V(V.nb_rows() + B.nb_rows(), V.nb_cols());
            // big_U c'est facile
            for (int i = 0; i < U.nb_rows; ++i) {
                std::copy(U.begin() + i * U.nb_cols(), U.begin() + (i + 1) * U.nb_cols(), big_U.begin() + i * (U.nb_cols() + A.nb_cols()));
                std::copy(A.begin() + i * A.nb_cols(), A.begin() + (i + 1) * A.nb_cols(), big_V.begin() + i * (U.nb_cols() + A.nb_cols()) + U.nb_cols());
            }
            // big_V , peut être en concaténant la transposé on peut raccourcir mais ca veut dire qu'il faut changer tot le code
            for (int l = 0; l < V.nb_cols(); ++l) {
                for (int k = 0; k < V.nb_rows(); ++k) {
                    big_V(k, l) = V(k, l);
                }
                for (int k = 0; k < B.nb_rows(); ++k) {
                    big_V(k + V.nb_rows(), l);
                }
            }
            // QR sur big_U
            int lda_u   = target_size;
            int lwork_u = 10 * target_size;
            int info_u;

            int N = U.nb_cols() + A.nb_cols();
            std::vector<double> work_u(lwork_u);
            std::vector<double> tau_u(N);
            Lapack<double>::geqrf(&target_size, &N, big_U.data(), &lda_u, tau_u.data(), work_u.data(), &lwork_u, &info_u);
            Matrix<double> R_u(N, N);
            for (int k = 0; k < N; ++k) {
                for (int l = k; l < N; ++l) {
                    R_u(k, l) = big_U(k, l);
                }
            }
            std::vector<double> workU(lwork_u);
            Lapack<double>::orgqr(&target_size, &N, &N, big_U.data(), &lda_u, tau_u.data(), workU.data(), &lwork_u, &info_u);

            // QR sur big_V    -------> je suis presque sure qu'il réécrit par dessus mais dans le doute on réinitialise
            int lda_v   = source_size;
            int lwork_v = 10 * source_size;
            int info_v;
            int M = V.nb_rows() + B.nb_rows(); // normelement c'est égale a N ;
            std::vector<double> work_v(lwork_v);
            std::vector<double> tau_v(M);
            Lapack<double>::geqrf(&M, &source_size, big_V.data(), &lda_v, tau_v.data(), work_v.data(), &lwork_v, &info_v);
            Matrix<double> R_v(M, M);
            for (int k = 0; k < M; ++k) {
                for (int l = k; l < M; ++l) {
                    R_v(k, l) = big_V(k, l);
                }
            }
            std::vector<double> workV(lwork_v);
            Lapack<CoefficientPrecision>::orgqr(&M, &source_size, &M, big_V.data(), &lda_v, tau_v.data(), workV.data(), &lwork_v, &info_v);

            // Ru*Rv
            Matrix<CoefficientPrecision> RuRv(N, N);
            R_u.add_matrix_product('N', 1.0, R_v.data(), 1.0, RuRv.data(), N);
            // truncated SVD sur R_uR_v
            std::vector<CoefficientPrecision> S(N, 0.0);
            Matrix<CoefficientPrecision> Uu(N, N);
            Matrix<CoefficientPrecision> VT(N, N);
            int info;
            int lwork = 10 * N;
            std::vector<CoefficientPrecision> work(lwork);
            Lapack<CoefficientPrecision>::gesvd('A', 'A', &N, &N, RuRv.data(), &N, S.data(), Uu.data(), &N, VT.data(), &N, work.data(), &lwork, &info);
            // truncation index
            int trunc = 0;
            if (rk == -1) {
                while (S[trunc] > epsilon) {
                    trunc += 1;
                }
            } else if (epsilon == -1) {
                trunc = rk;
            } else {
                std::cout << "il faut un critère rank ou epsilon , pas encore implémenter rk && epsilon" << std::endl;
            }
            // on tronque U S et V
            Matrix<CoefficientPrecision> Strunc(trunc, trunc);
            for (int k = 0; k < trunc; ++k) {
                Strunc(k, k) = S[k];
            }
            Matrix<CoefficientPrecision> Utrunc(target_size, trunc);
            Matrix<CoefficientPrecision> Vtrunc(trunc, source_size);
            for (int k = 0; k < target_size; ++k) {
                for (int l = 0; l < trunc; ++l) {
                    Utrunc(k, l) = Uu(k, l);
                }
            }
            for (int l = 0; l < source_size; ++l) {
                for (int k = 0; k < trunc; ++k) {
                    Vtrunc(k, l) = VT(k, l);
                }
            }
            // U = big_U Utrunc Strunc     --------------   V= VT big_V ;
            Matrix<CoefficientPrecision> temp(target_size, trunc);
            big_U.add_matrix_product('N', 1.0, Utrunc.data(), 1.0, temp.data(), &trunc);
            temp.add_matrix_product('N', 1.0, Strunc.data(), 0.0, temp.data(), &trunc);
            VT.add_matrix_product('N', 1.0, big_V.data(), 0.0, VT.data(), &source_size);
            //////
            Sr.push_back(temp);
            Sr.push_back(VT);
        }
    }
    //////////////////////
    //// ACA avec sum expression: A = UV
    /// --------> pour le pivot
    std::vector<Matrix<CoefficientPrecision>> ACA(const int &rkmax, const double &epsilon) const {
        std::vector<Matrix<CoefficientPrecision>> res;
        std::vector<std::vector<CoefficientPrecision>> col_U, row_V;
        int jr  = -1;
        int M   = target_size;
        int N   = source_size;
        int inc = 1;
        // first column --------------> we take j0 = 0 ;
        auto col = this->get_col(0);
        // first row pivot
        auto ir = get_pivot(col, -1);
        if (ir == -1) {
            std::cout << "corner case" << std::endl;
            Matrix<CoefficientPrecision> unull(1, 1);
            Matrix<CoefficientPrecision> vnull(1, 1);
            res.push_back(unull);
            res.push_back(vnull);
            return res;
        }
        auto delta = col[ir];
        // std::cout << "pivot ???" << ir << std::endl;
        std::vector<CoefficientPrecision> scaled(col.size(), 0.0);
        CoefficientPrecision alpha = 1.0 / delta;
        // std::cout << "alpha ??? " << alpha << std::endl;
        // Blas<CoefficientPrecision>::axpy(&source_size, &alpha, col.data(), &inc, scaled.data(), &inc);
        for (int k = 0; k < M; ++k) {
            col[k] = col[k] * (1.0 / delta);
        }
        // scale column
        col_U.push_back(col);
        // first row
        auto row = this->get_row(ir);
        row_V.push_back(row);
        // double error = 100000;
        int rk = 1;
        int R;
        // if (rkmax == -1) {
        //     R = (target_size + source_size) / 2;
        // } else {
        //     R = rkmax;
        // }
        CoefficientPrecision frob = 0;
        CoefficientPrecision aux  = 0;
        auto e1                   = norm2(scaled) * norm2(row);
        // && (error > e1 * epsilon)
        CoefficientPrecision error = 10000.0;
        std::cout << "____________________________________" << std::endl;
        while ((rk < rkmax) && (error > epsilon)) {
            std::vector<CoefficientPrecision> previous_row = row_V[rk - 1];
            // on doit pas prendre de pivot identique au précédent , on initialise jr a -1 pour passer le premier cas
            auto new_jr = get_pivot(previous_row, jr);
            if (new_jr == -1.0) {
                std::cout << "0 before suitable approx" << std::endl;
                Matrix<CoefficientPrecision> unull(1, 1);
                Matrix<CoefficientPrecision> vnull(1, 1);
                res.push_back(unull);
                res.push_back(vnull);
                return res;
            }
            jr = new_jr;
            // current column
            std::vector<CoefficientPrecision> current_col = this->get_col(jr);
            // Uk = col(jr) - sum ( Ui Vi[jr])
            for (int k = 0; k < rk; ++k) {
                double alpha = -1.0 * row_V[k][jr];
                auto col_k   = col_U[k];
                Blas<CoefficientPrecision>::axpy(&N, &alpha, col_k.data(), &inc, current_col.data(), &inc);
            }
            // next row pivot --------> Bebbendorf : il peut il y avoir des soucis mais a prioiri pas besoin de cheker si il apparait déja
            ir = get_pivot(current_col, -1);
            if (ir == -1.0) {
                std::cout << "0 before suitable approx" << std::endl;
                Matrix<CoefficientPrecision> unull(1, 1);
                Matrix<CoefficientPrecision> vnull(1, 1);
                res.push_back(unull);
                res.push_back(vnull);
                return res;
            }
            delta = current_col[ir];
            // scaling of the column
            for (int k = 0; k < M; ++k) {
                current_col[k] = current_col[k] * (1.0 / delta);
            }
            // std::vector<CoefficientPrecision> current_scaled(current_col.size());
            // alpha = 1.0 / delta;
            // Blas<CoefficientPrecision>::axpy(&N, &alpha, current_col.data(), &inc, scaled.data(), &inc);
            col_U.push_back(current_col);
            //// row
            auto current_row = this->get_row(ir);
            // Vk = row(ir) - sum( Vj Uj[ir])
            for (int k = 0; k < rk; ++k) {
                double alpha = -1.0 * col_U[k][ir];
                auto row_k   = row_V[k];
                Blas<double>::axpy(&N, &alpha, row_k.data(), &inc, current_row.data(), &inc);
            }
            row_V.push_back(current_row);
            CoefficientPrecision frob_aux = 0;
            aux                           = norm2(current_col) * norm2(current_row);
            for (int k = 0; k < rk; ++k) {
                auto col = col_U[k];
                auto row = row_V[k];
                frob_aux += std::abs(Blas<CoefficientPrecision>::dot(&M, current_col.data(), &inc, row.data(), &inc) * Blas<CoefficientPrecision>::dot(&N, current_row.data(), &(inc), col.data(), &(inc)));
            }

            frob += aux + 2 * std::real(frob_aux);
            // error = norm2(current_col) * norm2(current_row);
            error = sqrt(aux / frob);
            std::cout << error << ',' << aux << ',' << frob << ',' << frob_aux << std::endl;
            rk += 1;
        }
        // std::cout << "R et rk " << R << ',' << rk << std::endl;
        // std::cout << "rk ========" << rk << std::endl;
        // std::cout << (target_size + source_size) / 2 << std::endl;
        if (rk > rkmax) {
            std::cout << "ACA didn't find any approximation" << std::endl;
            Matrix<CoefficientPrecision> unull(1, 1);
            Matrix<CoefficientPrecision> vnull(1, 1);
            res.push_back(unull);
            res.push_back(vnull);
        } else {
            // asemblage des matrices en concaténant les liugnes et les colonnes
            // std::cout << "-----------> size" << col_U.size() << ',' << row_V.size() << std::endl;
            Matrix<CoefficientPrecision> U(target_size, rk);
            Matrix<CoefficientPrecision> V(rk, source_size);
            for (int l = 0; l < rk; ++l) {
                auto col = col_U[l];
                for (int k = 0; k < target_size; ++k) {
                    U(k, l) = col[k];
                }
                auto row = row_V[l];
                for (int k = 0; k < source_size; ++k) {
                    V(l, k) = row[k];
                }
            }

            res.push_back(U);
            res.push_back(V);
        }
        return res;
    }
    ////////////////////
    /// Pas mal mais il reste toujours des gros blocs dense qui nous frocent a tout densifier
    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override { // on regarde si il faut pas juste une ligne ou une collonne
        Matrix<CoefficientPrecision> mat(M, N);
        mat.assign(M, N, ptr, false);
        int ref_row = row_offset - target_offset;
        int ref_col = col_offset - source_offset;
        if (M == 1) {
            auto row = this->get_row(ref_row);
            if (N == source_size) {
                std::copy(row.begin(), row.begin() + N, ptr);

            } else {
                for (int l = 0; l < N; ++l) {
                    ptr[l] = row[l + ref_col];
                }
            }
        } else if (N == 1) {
            auto col = this->get_col(ref_col);
            if (M == target_size) {
                std::copy(col.begin(), col.begin() + M, ptr);
                // ptr = col.data();
                // std::cout << normFrob(mat) << std::endl;

            } else {
                for (int k = 0; k < M; ++k) {
                    mat(k, 0) = col[k + ref_row];
                }
            }
        }

        else {
            // donc la on veut un bloc, on va faire un choix arbitraire pour appeller copy_to_dense
            if ((M + N) > (target_size + source_size) / 2) {
                // std::cout << "énorme bloc densifié : " << M << ',' << N << " dans un bloc de taille :  " << target_size << ',' << source_size << "|" << target_offset << ',' << source_offset << std::endl;

                for (int k = 0; k < Sh.size() / 2; ++k) {
                    auto &H = Sh[2 * k];
                    auto &K = Sh[2 * k + 1];
                    Matrix<CoefficientPrecision> hdense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                    Matrix<CoefficientPrecision> kdense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                    copy_to_dense(*H, hdense.data());
                    copy_to_dense(*K, kdense.data());
                    // hdense.add_matrix_product('N', 1.0, kdense.data(), 0.0, hdense.data(), hdense.nb_cols());
                    Matrix<CoefficientPrecision> hk = hdense * kdense;
                    for (int i = 0; i < M; ++i) {
                        for (int j = 0; j < N; ++j) {
                            // mat(i, j) += hk(i + row_offset - H->get_target_cluster().get_offset(), j + col_offset - K->get_source_cluster().get_offset());
                            mat(i, j) += hk(i + row_offset - target_offset, j + col_offset - source_offset);
                            // mat(i, j) += hdense(i + row_offset - target_offset, j + col_offset - source_offset);
                        }
                    }
                }
                for (int k = 0; k < Sr.size() / 2; ++k) {
                    auto U = Sr[2 * k];
                    auto V = Sr[2 * k + 1];
                    Matrix<CoefficientPrecision> U_restr(target_size, U.nb_cols());
                    Matrix<CoefficientPrecision> V_restr(V.nb_rows(), source_size);
                    for (int i = 0; i < target_size; ++i) {
                        U_restr.set_row(i, U.get_row(i + target_offset - offset[2 * k]));
                    }
                    for (int j = 0; j < source_size; ++j) {
                        V_restr.set_col(j, V.get_col(j + source_offset - offset[2 * k + 1]));
                    }
                    auto uv = U_restr * V_restr;
                    for (int i = 0; i < M; ++i) {
                        for (int j = 0; j < N; ++j) {
                            mat(i, j) += uv(i + row_offset - target_offset, j + col_offset - source_offset);
                        }
                    }
                }

            } else {
                for (int k = 0; k < M; ++k) {
                    auto row = this->get_row(ref_row + k);
                    for (int l = 0; l < N; ++l) {
                        mat(k, l) = row[l + ref_col];
                    }
                }
            }
        }
    }

    // void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
    //     Matrix<CoefficientPrecision> mat(M, N);
    //     mat.assign(M, N, ptr, false);
    //     for (int k = 0; k < Sh.size() / 2; ++k) {
    //         auto &H = Sh[2 * k];
    //         auto &K = Sh[2 * k + 1];
    //         Matrix<CoefficientPrecision> hdense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
    //         Matrix<CoefficientPrecision> kdense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
    //         copy_to_dense(*H, hdense.data());
    //         copy_to_dense(*K, kdense.data());

    //         Matrix<CoefficientPrecision> hk = hdense * kdense;
    //         for (int i = 0; i < M; ++i) {
    //             for (int j = 0; j < N; ++j) {
    //                 // mat(i, j) += hk(i + row_offset - H->get_target_cluster().get_offset(), j + col_offset - K->get_source_cluster().get_offset());
    //                 mat(i, j) += hk(i + row_offset - target_offset, j + col_offset - source_offset);
    //             }
    //         }
    //     }
    //     for (int k = 0; k < Sr.size() / 2; ++k) {
    //         auto U = Sr[2 * k];
    //         auto V = Sr[2 * k + 1];
    //         Matrix<CoefficientPrecision> U_restr(target_size, U.nb_cols());
    //         Matrix<CoefficientPrecision> V_restr(V.nb_rows(), source_size);
    //         for (int i = 0; i < target_size; ++i) {
    //             U_restr.set_row(i, U.get_row(i + target_offset - offset[2 * k]));
    //         }
    //         for (int j = 0; j < source_size; ++j) {
    //             V_restr.set_col(j, V.get_col(j + source_offset - offset[2 * k + 1]));
    //         }
    //         auto uv = U_restr * V_restr;
    //         for (int i = 0; i < M; ++i) {
    //             for (int j = 0; j < N; ++j) {
    //                 mat(i, j) += uv(i + row_offset - target_offset, j + col_offset - source_offset);
    //             }
    //         }
    //     }
    // }

    const std::vector<int>
    is_restrictible() const {
        std::vector<int> test(2, 0.0);
        int k      = 0;
        bool test1 = true;
        bool test2 = true;

        while (k < Sh.size() / 2 and (test1 and test2)) {
            auto H = Sh[2 * k];
            auto K = Sh[2 * k + 1];
            if (H->get_children().size() == 0) {
                test[0] = 1;
                test1   = true;
            }
            if (K->get_children().size() == 0) {
                test[1] = 1;
                test2   = true;
            }
            k += 1;
        }
        return test;
    }
    bool is_restr() const {
        return restrictible;
    }

    ///////////////////////////////////////////////////////////
    ///// Random ACA -> sans vitual generator
    /////////////////////////////////////////////////////

    //     const LowRankMatrix<CoefficientPrecision, CoordinatePrecision> rd_ACA(const int &res_rk, const CoefficientPrecision &eps) {
    //         Matrix<CoefficientPrecision> Id(target_size, source_size);
    //         Matrix<CoefficientPrecision> L(target_size, req_rk);
    //         Matrix<CoefficientPrecision> Lt(req_rk, target_size);
    //         for (int k = 0; k < target_size; ++k) {
    //             Id(k, k) = 1.0;
    //         }
    //         for (int k = 0; k < req_rk; ++k) {
    //             std::vector<CoefficientPrecision> w(target_size, 0.0);
    //             std::random_device rd;
    //             std::mt19937 gen(rd());
    //             std::normal_distribution<double> dis(0, 1.0);
    //             // utiliser std::generate pour remplir le vecteur avec des nombres aléatoires
    //             std::generate(w.begin(), w.end(), [&]() { return dis(gen); });

    //             std::vector<CoefficientPrecision> col = (Id - L * Lt) * this->new_prod(w);
    //             for (int i = 0; i < target_size; ++i) {
    //                 L(i, k)  = w[i];
    //                 Lt(k, i) = w[i];
    //             }
    //         }

    // }

    // ///////////////////////////////////////////////////////
    // // RESTRICT
    // /////////////////////////////////////////////

    ////////////////////
    // Arthur : le test_restrict est trés lourd , on pourrait utiliser get_block MAIS il faudrait pouvoir renvoyer soit une hmat soit une mat
    // ( cf commentaire de get_block()) pour voire les soucis
    ////////
    const SumExpression
    Restrict(int target_size0, int target_offset0, int source_size0, int source_offset0) const {

        std::vector<const HMatrixType *> sh;
        auto of = offset;
        auto sr = Sr;
        for (int rep = 0; rep < Sh.size() / 2; ++rep) {
            auto &H          = Sh[2 * rep];
            auto &K          = Sh[2 * rep + 1];
            auto &H_children = H->get_children();
            auto &K_children = K->get_children();

            if ((H_children.size() > 0) and (K_children.size() > 0)) {
                for (auto &H_child : H_children) {
                    if ((H_child->get_target_cluster().get_size() == target_size0) and (H_child->get_target_cluster().get_offset() == target_offset0)) {
                        for (auto &K_child : K_children) {
                            if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
                                if ((K_child->get_target_cluster().get_size() == H_child->get_source_cluster().get_size())
                                    and (K_child->get_target_cluster().get_offset() == H_child->get_source_cluster().get_offset())) {
                                    if (H_child->is_low_rank() or K_child->is_low_rank()) {
                                        if (H_child->is_low_rank() and K_child->is_low_rank()) {

                                            Matrix<CoefficientPrecision> uh = H_child->get_low_rank_data()->Get_U();
                                            Matrix<CoefficientPrecision> vh = H_child->get_low_rank_data()->Get_V();
                                            Matrix<CoefficientPrecision> uk = K_child->get_low_rank_data()->Get_U();
                                            Matrix<CoefficientPrecision> vk = K_child->get_low_rank_data()->Get_V();
                                            Matrix<CoefficientPrecision> v  = vh * uk * vk;
                                            sr.push_back(uh);
                                            sr.push_back(v);
                                            of.push_back(H_child->get_target_cluster().get_offset());
                                            of.push_back(K_child->get_source_cluster().get_offset());
                                        } else if ((H_child->is_low_rank()) and !(K_child->is_low_rank())) {

                                            Matrix<CoefficientPrecision> u = H_child->get_low_rank_data()->Get_U();
                                            Matrix<CoefficientPrecision> v = H_child->get_low_rank_data()->Get_V();
                                            // Matrix<CoefficientPrecision> p = K_child->lr_hmat(v);

                                            auto v_transp = v.transp(v);
                                            std::vector<CoefficientPrecision> v_row_major(v.nb_cols() * v.nb_rows());
                                            for (int k = 0; k < v.nb_rows(); ++k) {
                                                for (int l = 0; l < v.nb_cols(); ++l) {
                                                    v_row_major[l * v.nb_rows() + k] = v_transp.data()[k * v.nb_cols() + l];
                                                }
                                            }

                                            Matrix<CoefficientPrecision> vk(K_child->get_source_cluster().get_size(), v.nb_rows());
                                            K_child->add_matrix_product_row_major('T', 1.0, v_row_major.data(), 0.0, vk.data(), v.nb_rows());
                                            Matrix<CoefficientPrecision> vK(K_child->get_source_cluster().get_size(), v.nb_rows());

                                            for (int k = 0; k < K_child->get_source_cluster().get_size(); ++k) {
                                                for (int l = 0; l < v.nb_rows(); ++l) {
                                                    vK.data()[l * K_child->get_source_cluster().get_size() + k] = vk.data()[k * v.nb_rows() + l];
                                                }
                                            }
                                            Matrix<CoefficientPrecision> p = vk.transp(vK);

                                            // std::cout << p.nb_rows() << ',' << p.nb_cols() << std::endl;
                                            // std::cout << v.nb_rows() << ',' << K_child->get_source_cluster().get_size() << std::endl;
                                            // std::cout << normFrob(p) << ',' << normFrob(vvK) << std::endl;
                                            // std::cout << normFrob(p - vvK) / normFrob(vvK) << std::endl;

                                            sr.push_back(u);
                                            sr.push_back(p);

                                            of.push_back(H_child->get_target_cluster().get_offset());
                                            of.push_back(K_child->get_source_cluster().get_offset());

                                        } else if (!(H_child->is_low_rank()) and (K_child->is_low_rank())) {

                                            // std::cout << "!3" << std::endl;
                                            Matrix<CoefficientPrecision> u = K_child->get_low_rank_data()->Get_U();
                                            Matrix<CoefficientPrecision> v = K_child->get_low_rank_data()->Get_V();
                                            // Matrix<CoefficientPrecision> Hu = H_child->hmat_lr(u);
                                            std::vector<CoefficientPrecision> u_row_major(u.nb_rows() * u.nb_cols(), 0.0);
                                            for (int k = 0; k < u.nb_rows(); ++k) {
                                                for (int l = 0; l < u.nb_cols(); ++l) {
                                                    u_row_major[k * u.nb_cols() + l] = u.data()[k + u.nb_rows() * l];
                                                }
                                            }
                                            Matrix<CoefficientPrecision> hu(H_child->get_target_cluster().get_size(), u.nb_cols());
                                            H_child->add_matrix_product_row_major('N', 1.0, u_row_major.data(), 0.0, hu.data(), u.nb_cols());
                                            Matrix<CoefficientPrecision> Hu(H_child->get_target_cluster().get_size(), u.nb_cols());
                                            for (int k = 0; k < H_child->get_target_cluster().get_size(); ++k) {
                                                for (int l = 0; l < u.nb_cols(); ++l) {
                                                    Hu.data()[l * H_child->get_target_cluster().get_size() + k] = hu.data()[k * u.nb_cols() + l];
                                                }
                                            }
                                            sr.push_back(Hu);
                                            sr.push_back(v);
                                            of.push_back(H_child->get_target_cluster().get_offset());
                                            of.push_back(K_child->get_source_cluster().get_offset());
                                        }
                                    } else {

                                        sh.push_back(H_child.get());
                                        sh.push_back(K_child.get());
                                    }
                                }
                            }
                        }
                    }
                }
            } else if (H_children.size() == 0 and K_children.size() > 0) {
                // H is dense otherwise HK would be in SR
                if (target_size0 < H->get_target_cluster().get_size()) {
                    std::cout << " mauvais restrict , bloc dense insécablble a gauche" << std::endl;

                } else {
                    // donc la normalement K a juste 2 fils et on prend celui qui a le bon sigma
                    for (int l = 0; l < K_children.size(); ++l) {
                        auto &K_child = K_children[l];
                        if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
                            // soit k_child est lr soit il l'est pas ;
                            if (K_child->is_low_rank()) {

                                Matrix<CoefficientPrecision> u = K_child->get_low_rank_data()->Get_U();
                                Matrix<CoefficientPrecision> v = K_child->get_low_rank_data()->Get_V();
                                // Matrix<CoefficientPrecision> Hu = H->hmat_lr(u);
                                std::vector<CoefficientPrecision> u_row_major(u.nb_rows() * u.nb_cols());
                                for (int k = 0; k < u.nb_rows(); ++k) {
                                    for (int kk = 0; kk < u.nb_cols(); ++kk) {
                                        u_row_major[k * u.nb_cols() + kk] = u.data()[k + kk * u.nb_rows()];
                                    }
                                }
                                Matrix<CoefficientPrecision> hu(H->get_target_cluster().get_size(), u.nb_cols());

                                H->add_matrix_product_row_major('N', 1.0, u_row_major.data(), 0.0, hu.data(), u.nb_cols());
                                Matrix<CoefficientPrecision> Hu(H->get_target_cluster().get_size(), u.nb_cols());
                                for (int k = 0; k < H->get_target_cluster().get_size(); ++k) {
                                    for (int kk = 0; kk < u.nb_cols(); ++kk) {
                                        Hu.data()[kk * H->get_target_cluster().get_size() + k] = hu.data()[k * u.nb_cols() + kk];
                                    }
                                }
                                sr.push_back(Hu);
                                sr.push_back(v);
                                of.push_back(H->get_target_cluster().get_offset());
                                of.push_back(K_child->get_source_cluster().get_offset());
                            } else {
                                sh.push_back(H);
                                sh.push_back(K_child.get());
                            }
                        }
                    }
                }
            } else if (H_children.size() > 0 and K_children.size() == 0) {
                // K is dense sinson ce serait dans sr
                if (source_size0 < K->get_source_cluster().get_size()) {
                    std::cout << "mauvais restric , bloc dense insécable à doite" << std::endl;
                } else {
                    for (int l = 0; l < H_children.size(); ++l) {
                        auto &H_child = H_children[l];
                        if (H_child->get_target_cluster().get_size() == target_size0 and H_child->get_target_cluster().get_offset() == target_offset0) {
                            if (H_child->is_low_rank()) {
                                std::cout << "cici" << std::endl;
                                Matrix<CoefficientPrecision> u = H_child->get_low_rank_data()->Get_U();
                                Matrix<CoefficientPrecision> v = H_child->get_low_rank_data()->Get_V();
                                // Matrix<CoefficientPrecision> vK = K->lr_hmat(v);
                                auto v_transp = v.transp(v);
                                std::vector<CoefficientPrecision> v_row_major(v.nb_rows(), v.nb_cols());
                                for (int k = 0; k < v.nb_rows(); ++k) {
                                    for (int kk = 0; kk < v.nb_cols(); +kk) {
                                        v_row_major[k * v.nb_cols() + kk] = v_transp.data()[k + kk * v.nb_rows()];
                                    }
                                }
                                Matrix<CoefficientPrecision> vk(K->get_source_cluster().get_size(), v.nb_rows());
                                K->add_matrix_product_row_major('T', 1.0, v_row_major.data(), 0.0, vk.data(), v.nb_rows());
                                Matrix<CoefficientPrecision> vK(v.nb_rows(), K->get_source_cluster().get_size());
                                for (int k = 0; k < v.nb_rows(); ++k) {
                                    for (int kk = 0; kk < K->get_source_cluster().get_size(); ++kk) {
                                        vK.data()[k * K->get_source_cluster().get_size() + kk] = vk.data()[k + kk * v.nb_rows()];
                                    }
                                }
                                sr.push_back(u);
                                sr.push_back(vK);

                                of.push_back(H_child->get_target_cluster().get_offset());
                                of.push_back(K->get_source_cluster().get_offset());
                            } else {
                                sh.push_back(H_child.get());
                                sh.push_back(K);
                            }
                        }
                    }
                }
            } else if ((H_children.size() == 0) and (K_children.size() == 0)) {
                // H and K are dense sinon elle serait dans sr
                // on peut pas les couper
                std::cout << " mauvaise restriction : matrice insécable a gauche et a droite" << std::endl;
            }
        }
        SumExpression res(sr, sh, of, target_size0, target_offset0, source_size0, source_offset0);
        // for (int k = 0; k < of.size(); ++k) {
        //     std::cout << of[k] << std::endl;
        // }
        return res;
    }

    const SumExpression Restrict_s(int target_0, int source_0, int target_offset0, int source_offset0) const {
        std::vector<const HMatrixType *> sh;
        auto of   = offset;
        auto sr   = Sr;
        bool flag = true;
        std::vector<int> mid_of;
        for (int k = 0; k < Sh.size() / 2; ++k) {
            auto &H = Sh[2 * k];
            auto &K = Sh[2 * k + 1];
            for (auto &rho_child : H->get_source_cluster().get_children()) {
                auto Htr = H->get_block(target_0, rho_child->get_size(), target_offset0, rho_child->get_offset());
                auto Krs = K->get_block(rho_child->get_size(), source_0, rho_child->get_offset(), source_offset0);
                // if ( (Htr->get_target_cluster().get_size() == target_0 )and (Htr->get_source_cluster().get_size()  == rho_child->get_size() ) and ( Krs->get_target_cluster().get_size() == rho_child->get_size() ) and ( Krs->get_source_cluster().get_size() == source_0) ){
                if ((!(Htr->get_target_cluster().get_size() == target_0) or !(Htr->get_source_cluster().get_size() == rho_child->get_size()) or !(Krs->get_target_cluster().get_size() == rho_child->get_size()) or !(Krs->get_source_cluster().get_size() == source_0))) {
                    // ca veut dire qu'il faut extraire les bons sous blocs
                    if ((Htr->is_low_rank()) or (Krs->is_low_rank())) {
                        if ((Htr->is_low_rank()) and (Krs->is_low_rank())) {
                            auto Uh = Htr->get_low_rank_data()->Get_U().get_block(target_0, Htr->get_low_rank_data()->rank_of(), target_offset0 - Htr->get_target_cluster().get_offset(), 0);
                            auto Vh = Htr->get_low_rank_data()->Get_V().get_block(Htr->get_low_rank_data()->rank_of(), rho_child->get_size(), 0, rho_child->get_offset() - Htr->get_source_cluster().get_offset());
                            auto Uk = Krs->get_low_rank_data()->Get_U().get_block(rho_child->get_size(), Krs->get_low_rank_data()->rank_of(), rho_child->get_offset() - Krs->get_target_cluster().get_offset(), 0);
                            auto Vk = Krs->get_low_rank_data()->Get_V().get_block(Krs->get_low_rank_data()->rank_of(), source_0, 0, source_offset0 - Krs->get_source_cluster().get_offset());
                            sr.push_back(Uh);
                            sr.push_back(Vh * Uk * Vk);
                            of.push_back(target_offset0);
                            of.push_back(source_offset0);
                        } else if ((Htr->is_low_rank()) and !(Krs->is_low_rank())) {
                            auto Uh = Htr->get_low_rank_data()->Get_U().get_block(target_0, Htr->get_low_rank_data()->rank_of(), target_offset0 - Htr->get_target_cluster().get_offset(), 0);
                            auto Vh = Htr->get_low_rank_data()->Get_V().get_block(Htr->get_low_rank_data()->rank_of(), rho_child->get_size(), 0, rho_child->get_offset() - Htr->get_source_cluster().get_offset());
                            if (Krs->get_target_cluster().get_size() == rho_child->get_size() && Krs->get_source_cluster().get_size() == source_0) {
                                sr.push_back(Uh);
                                sr.push_back(Krs->matrix_hmatrix(Vh));
                                of.push_back(target_offset0);
                                of.push_back(source_offset0);
                            } else { // Krs est pas low rank et n'a pas la bonne taille-> ca devrait pas arriver mais c'est qu'il y a un gros blocs dense
                                auto krestr = Krs->get_dense_data()->get_block(rho_child->get_size(), source_0, rho_child->get_offset() - Krs->get_target_cluster().get_offset(), source_offset0 - Krs->get_source_cluster().get_offset());
                                sr.push_back(Uh);
                                sr.push_back(Vh * krestr);
                                of.push_back(target_offset0);
                                of.push_back(source_offset0);
                            }
                        } else if (!(Htr->is_low_rank()) and (Krs->is_low_rank())) {
                            auto Uk = Krs->get_low_rank_data()->Get_U().get_block(rho_child->get_size(), Krs->get_low_rank_data()->rank_of(), source_offset0 - Krs->get_target_cluster().get_offset(), 0);
                            auto Vk = Krs->get_low_rank_data()->Get_V().get_block(Krs->get_low_rank_data()->rank_of(), source_0, 0, source_offset0 - Krs->get_source_cluster().get_offset());
                            if (Htr->get_source_cluster().get_size() == rho_child->get_size() && Htr->get_target_cluster().get_size() == target_0) {
                                sr.push_back(Htr->hmatrix_matrix(Uk));
                                sr.push_back(Vk);
                                of.push_back(target_offset0);
                                of.push_back(source_offset0);
                            } else { // Htr est pas low rank et n'a pas la bonne taille-> ca devrait pas arriver mais c'est qu'il y a un gros blocs dense
                                auto Hrestr = Htr->get_dense_data()->get_block(target_0, rho_child->get_size(), target_offset0 - Htr->get_target_cluster().get_offset(), rho_child->get_offset() - Htr->get_source_cluster().get_offset());
                                sr.push_back(Hrestr * Uk);
                                sr.push_back(Vk);
                                of.push_back(target_offset0);
                                of.push_back(source_offset0);
                            }
                        } else {
                            std::cout << "?!?!?!?!" << std::endl;
                        }
                    }
                } else {

                    if (!(Htr->is_low_rank()) and !(Krs->is_low_rank())) {
                        sh.push_back(Htr);
                        sh.push_back(Krs);
                        // mid_of.push_back(rho_child->get_size());
                        // mid_of.push_back(rho_child->get_offset());
                        if (Htr->get_children().size() == 0 && Krs->get_children().size() == 0) {
                            flag = false;
                        }
                    } else {
                        if ((Htr->is_low_rank()) and !(Krs->is_low_rank())) {
                            auto U = Htr->get_low_rank_data()->Get_U();
                            // auto V = Krs->mat_hmat(Htr->get_low_rank_data()->Get_V());
                            auto V = Krs->matrix_hmatrix(Htr->get_low_rank_data()->Get_V());

                            sr.push_back(U);
                            sr.push_back(V);
                            of.push_back(target_offset0);
                            of.push_back(source_offset0);
                        } else if (!(Htr->is_low_rank()) and (Krs->is_low_rank())) {
                            // auto U = Htr->hmat_mat(Krs->get_low_rank_data()->Get_U());
                            auto U = Htr->hmatrix_matrix(Krs->get_low_rank_data()->Get_U());

                            auto V = Krs->get_low_rank_data()->Get_V();
                            sr.push_back(U);
                            sr.push_back(V);
                            of.push_back(target_offset0);
                            of.push_back(source_offset0);
                        } else if ((Htr->is_low_rank()) and (Krs->is_low_rank())) {
                            if (Htr->get_source_cluster().get_size() == Krs->get_target_cluster().get_size()) {

                                auto U = (Htr->get_low_rank_data()->Get_U() * Htr->get_low_rank_data()->Get_V()) * Krs->get_low_rank_data()->Get_U();
                                auto V = Krs->get_low_rank_data()->Get_V();
                                sr.push_back(U);
                                sr.push_back(V);
                                of.push_back(target_offset0);
                                of.push_back(source_offset0);
                            } else {
                                auto Uh = Htr->get_low_rank_data()->Get_U();
                                auto Vh = Htr->get_low_rank_data()->Get_V();
                                auto Uk = Krs->get_low_rank_data()->Get_U();
                                auto Vk = Krs->get_low_rank_data()->Get_V();
                                Matrix<CoefficientPrecision> urestrh(target_0, Uh.nb_cols());
                                Matrix<CoefficientPrecision> vrestrh(Uh.nb_cols(), rho_child->get_size());
                                Matrix<CoefficientPrecision> urestrk(rho_child->get_size(), Uk.nb_cols());
                                Matrix<CoefficientPrecision> vrestrk(Uk.nb_cols(), source_0);
                                if (Uh.nb_rows() == target_0) {
                                    urestrh = Uh;
                                } else {
                                    for (int k = 0; k < target_0; ++k) {
                                        for (int l = 0; l < Uh.nb_cols(); ++l) {
                                            urestrh(k, l) = Uh(k + target_offset0 - Htr->get_target_cluster().get_offset(), l);
                                        }
                                    }
                                }
                                if (Vh.nb_cols() == rho_child->get_size()) {
                                    vrestrh = Vh;
                                } else {
                                    for (int k = 0; k < Vh.nb_rows(); ++k) {
                                        for (int l = 0; l < rho_child->get_size(); ++l) {
                                            vrestrh(k, l) = Vh(k, l + rho_child->get_offset() - Htr->get_source_cluster().get_offset());
                                        }
                                    }
                                }
                                if (Uk.nb_rows() == rho_child->get_size()) {
                                    urestrk = Uk;
                                } else {
                                    for (int k = 0; k < rho_child->get_size(); ++k) {
                                        for (int l = 0; l < Uk.nb_cols(); ++l) {
                                            urestrk(k, l) = Uk(k + rho_child->get_offset() - Krs->get_source_cluster().get_offset(), l);
                                        }
                                    }
                                }
                                if (Vk.nb_cols() == source_0) {
                                    vrestrk = Vk;
                                } else {
                                    for (int k = 0; k < Vk.nb_rows(); ++k) {
                                        for (int l = 0; l < source_0; ++l) {
                                            vrestrk(k, l) = Vk(k, l + target_offset0 - Krs->get_source_cluster().get_offset());
                                        }
                                    }
                                }
                                auto U = urestrh * vrestrh * urestrk;
                                auto V = vrestrk;
                                sr.push_back(U);
                                sr.push_back(V);
                                of.push_back(target_offset0);
                                of.push_back(source_offset0);
                            }
                        }
                    }
                }
            }
        }
        SumExpression res(sr, sh, of, mid_of, target_0, target_offset0, source_0, source_offset0, flag);
        return res;
    }
    /////////////////////////
    //// restrcit "clean"
    //// TEST PASSED : YES -> erreur relative hmat hmat = e-5 , 73% de rang faible
    const SumExpression
    Restrict_clean(int target_size0, int target_offset0, int source_size0, int source_offset0) const {

        std::vector<const HMatrixType *> sh;
        auto of   = offset;
        auto sr   = Sr;
        bool flag = true;
        for (int rep = 0; rep < Sh.size() / 2; ++rep) {
            auto &H          = Sh[2 * rep];
            auto &K          = Sh[2 * rep + 1];
            auto &H_children = H->get_children();
            auto &K_children = K->get_children();
            ////////////////////////////
            // Je rajoute ca au cas ou il y a un grand bloc de zero , si un des deux est nul le produit est nul et ca sert a rien de garder les vrai blocs
            ///////////////////////////

            if ((H_children.size() > 0) and (K_children.size() > 0)) {
                for (auto &H_child : H_children) {
                    if ((H_child->get_target_cluster().get_size() == target_size0) and (H_child->get_target_cluster().get_offset() == target_offset0)) {
                        for (auto &K_child : K_children) {
                            if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
                                if ((K_child->get_target_cluster().get_size() == H_child->get_source_cluster().get_size())
                                    and (K_child->get_target_cluster().get_offset() == H_child->get_source_cluster().get_offset())) {
                                    if (H_child->is_low_rank() or K_child->is_low_rank()) {
                                        if (H_child->is_low_rank() and K_child->is_low_rank()) {
                                            auto v1 = H_child->get_low_rank_data()->Get_V();
                                            auto v2 = K_child->get_low_rank_data()->Get_U();
                                            auto v3 = K_child->get_low_rank_data()->Get_V();

                                            auto v = H_child->get_low_rank_data()->Get_V() * K_child->get_low_rank_data()->Get_U() * K_child->get_low_rank_data()->Get_V();
                                            sr.push_back(H_child->get_low_rank_data()->Get_U());
                                            sr.push_back(v);
                                            of.push_back(H_child->get_target_cluster().get_offset());
                                            of.push_back(K_child->get_source_cluster().get_offset());
                                        } else if ((H_child->is_low_rank()) and !(K_child->is_low_rank())) {
                                            auto vK = K_child->mat_hmat(H_child->get_low_rank_data()->Get_V());
                                            // auto vK = K_child->matrix_hmatrix(H_child->get_low_rank_data()->Get_V());

                                            sr.push_back(H_child->get_low_rank_data()->Get_U());
                                            sr.push_back(vK);
                                            of.push_back(H_child->get_target_cluster().get_offset());
                                            of.push_back(K_child->get_source_cluster().get_offset());
                                        } else if (!(H_child->is_low_rank()) and (K_child->is_low_rank())) {
                                            auto Hu = H_child->hmat_mat(K_child->get_low_rank_data()->Get_U());
                                            // auto Hu = H_child->hmatrix_matrix(K_child->get_low_rank_data()->Get_U());

                                            sr.push_back(Hu);
                                            sr.push_back(K_child->get_low_rank_data()->Get_V());
                                            of.push_back(H_child->get_target_cluster().get_offset());
                                            of.push_back(K_child->get_source_cluster().get_offset());
                                        }
                                    } else {
                                        sh.push_back(H_child.get());
                                        sh.push_back(K_child.get());
                                        // if (H_child->get_children().size == 0 && K_child->get_children().size() == 0) {
                                        //     flag = false;
                                        // }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // } else {
            //     std::cout << "cas pas possible ->  soit tout les fils soit aucun" << std::endl;
            // }
            else if (H_children.size() == 0 and K_children.size() > 0) {
                if (target_size0 < H->get_target_cluster().get_size()) {
                    std::cout << " mauvais restrict , bloc dense insécablble a gauche" << std::endl;
                } else {
                    for (int l = 0; l < K_children.size(); ++l) {
                        auto &K_child = K_children[l];
                        if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
                            if (K_child->is_low_rank()) {
                                // auto Hu = H->hmat_mat(K_child->get_low_rank_data()->Get_U());
                                auto Hu = H->hmatrix_matrix(K_child->get_low_rank_data()->Get_U());
                                sr.push_back(Hu);
                                sr.push_back(K_child->get_low_rank_data()->Get_V());
                                of.push_back(H->get_target_cluster().get_offset());
                                of.push_back(K_child->get_source_cluster().get_offset());
                            } else {
                                sh.push_back(H);
                                sh.push_back(K_child.get());
                            }
                        }
                    }
                }
            } else if (H_children.size() > 0 and K_children.size() == 0) {
                if (source_size0 < K->get_source_cluster().get_size()) {
                    std::cout << "mauvais restric , bloc dense insécable à doite" << std::endl;
                } else {
                    for (int l = 0; l < H_children.size(); ++l) {
                        auto &H_child = H_children[l];
                        if (H_child->get_target_cluster().get_size() == target_size0 and H_child->get_target_cluster().get_offset() == target_offset0) {
                            if (H_child->is_low_rank()) {
                                // auto vK = K->mat_hmat(H_child->get_low_rank_data()->Get_V());
                                auto vK = K->matrix_hmatrix(H_child->get_low_rank_data()->Get_V());

                                sr.push_back(H_child->get_low_rank_data()->Get_U());
                                sr.push_back(vK);

                                of.push_back(H_child->get_target_cluster().get_offset());
                                of.push_back(K->get_source_cluster().get_offset());
                            } else {
                                sh.push_back(H_child.get());
                                sh.push_back(K);
                            }
                        }
                    }
                }
            } else if ((H_children.size() == 0) and (K_children.size() == 0)) {
                std::cout << " mauvaise restriction : matrice insécable a gauche et a droite" << std::endl;
            }
        }

        SumExpression res(sr, sh, of, target_size0, target_offset0, source_size0, source_offset0, flag);
        return res;
    }

    const SumExpression
    Restrict_clean_acumulate_update(int target_size0, int target_offset0, int source_size0, int source_offset0) const {

        std::vector<const HMatrixType *> sh;
        auto of = offset;
        // on recopie UV
        std::vector<Matrix<CoefficientPrecision>> sr;
        bool flag = true;
        if (Sr.size() > 0) {
            // on restreint
            auto U = Sr[0];
            auto V = Sr[1];
            Matrix<CoefficientPrecision> urestr(target_size0, U.nb_cols());
            Matrix<CoefficientPrecision> vrestr(V.nb_rows(), source_size0);
            for (int k = 0; k < target_size; ++k) {
                for (int l = 0; l < U.nb_cols(); ++l) {
                    urestr(k, l) = U(k + target_offset0 - target_offset, l);
                }
            }
            for (int k = 0; k < V.nb_rows(); ++k) {
                for (int l = 0; l < source_size0; ++l) {
                    vrestr(k, l) = V(k, l + source_offset0 - source_offset);
                }
            }
            sr.push_back(urestr);
            sr.push_back(vrestr);
        }
        SumExpression res(sr, sh, of, target_size0, target_offset0, source_size0, source_offset0, flag);
        for (int rep = 0; rep < Sh.size() / 2; ++rep) {
            auto &H          = Sh[2 * rep];
            auto &K          = Sh[2 * rep + 1];
            auto &H_children = H->get_children();
            auto &K_children = K->get_children();
            ////////////////////////////
            // Je rajoute ca au cas ou il y a un grand bloc de zero , si un des deux est nul le produit est nul et ca sert a rien de garder les vrai blocs
            ///////////////////////////

            if ((H_children.size() > 0) and (K_children.size() > 0)) {
                for (auto &H_child : H_children) {
                    if ((H_child->get_target_cluster().get_size() == target_size0) and (H_child->get_target_cluster().get_offset() == target_offset0)) {
                        for (auto &K_child : K_children) {
                            if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
                                if ((K_child->get_target_cluster().get_size() == H_child->get_source_cluster().get_size())
                                    and (K_child->get_target_cluster().get_offset() == H_child->get_source_cluster().get_offset())) {
                                    if (H_child->is_low_rank() or K_child->is_low_rank()) {
                                        if (H_child->is_low_rank() and K_child->is_low_rank()) {
                                            auto v = H_child->get_low_rank_data()->Get_V();
                                            auto n = H_child->get_low_rank_data()->Get_U().nb_cols();
                                            Matrix<CoefficientPrecision> tempv(n, n);
                                            v.add_matrix_product('N', 1.0, K_child->get_low_rank_data()->Get_U().data(), 0.0, tempv.data(), v.nb_cols());
                                            Matrix<CoefficientPrecision> vend(n, source_size0);
                                            tempv.add_matrix_product('N', 1.0, K_child->get_low_rank_data()->Get_V().data(), 0.0, vend.data(), K_child->get_low_rank_data()->Get_V().nb_cols());
                                            res.actualise(H_child->get_low_rank_data()->Get_U(), vend);
                                            // auto v = H_child->get_low_rank_data()->Get_V() * K_child->get_low_rank_data()->Get_U() * K_child->get_low_rank_data()->Get_V();
                                            // sr.push_back(H_child->get_low_rank_data()->Get_U());
                                            // sr.push_back(v);
                                            // of.push_back(H_child->get_target_cluster().get_offset());
                                            // of.push_back(K_child->get_source_cluster().get_offset());
                                        } else if ((H_child->is_low_rank()) and !(K_child->is_low_rank())) {
                                            auto vK = K_child->mat_hmat(H_child->get_low_rank_data()->Get_V());
                                            res.actualise(H_child->get_low_rank_data()->Get_U(), vK);
                                            // auto vK = K_child->matrix_hmatrix(H_child->get_low_rank_data()->Get_V());

                                            // sr.push_back(H_child->get_low_rank_data()->Get_U());
                                            // sr.push_back(vK);
                                            // of.push_back(H_child->get_target_cluster().get_offset());
                                            // of.push_back(K_child->get_source_cluster().get_offset());
                                        } else if (!(H_child->is_low_rank()) and (K_child->is_low_rank())) {
                                            auto Hu = H_child->hmat_mat(K_child->get_low_rank_data()->Get_U());
                                            res.actualise(Hu, K_child->get_low_rank_data()->Get_V());
                                            // auto Hu = H_child->hmatrix_matrix(K_child->get_low_rank_data()->Get_U());

                                            // sr.push_back(Hu);
                                            // sr.push_back(K_child->get_low_rank_data()->Get_V());
                                            // of.push_back(H_child->get_target_cluster().get_offset());
                                            // of.push_back(K_child->get_source_cluster().get_offset());
                                        }
                                    } else {
                                        res.add_HK(H_child.get(), K_child.get());
                                        // sh.push_back(H_child.get());
                                        // sh.push_back(K_child.get());
                                        // if (H_child->get_children().size == 0 && K_child->get_children().size() == 0) {
                                        //     flag = false;
                                        // }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // }
            else {
                std::cout << "cas pas possible ->  soit tout les fils soit aucun" << std::endl;
            }
            // else if (H_children.size() == 0 and K_children.size() > 0) {
            //     if (target_size0 < H->get_target_cluster().get_size()) {
            //         std::cout << " mauvais restrict , bloc dense insécablble a gauche" << std::endl;
            //     } else {
            //         for (int l = 0; l < K_children.size(); ++l) {
            //             auto &K_child = K_children[l];
            //             if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
            //                 if (K_child->is_low_rank()) {
            //                     // auto Hu = H->hmat_mat(K_child->get_low_rank_data()->Get_U());
            //                     auto Hu = H->hmatrix_matrix(K_child->get_low_rank_data()->Get_U());
            //                     sr.push_back(Hu);
            //                     sr.push_back(K_child->get_low_rank_data()->Get_V());
            //                     of.push_back(H->get_target_cluster().get_offset());
            //                     of.push_back(K_child->get_source_cluster().get_offset());
            //                 } else {
            //                     sh.push_back(H);
            //                     sh.push_back(K_child.get());
            //                 }
            //             }
            //         }
            //     }
            // } else if (H_children.size() > 0 and K_children.size() == 0) {
            //     if (source_size0 < K->get_source_cluster().get_size()) {
            //         std::cout << "mauvais restric , bloc dense insécable à doite" << std::endl;
            //     } else {
            //         for (int l = 0; l < H_children.size(); ++l) {
            //             auto &H_child = H_children[l];
            //             if (H_child->get_target_cluster().get_size() == target_size0 and H_child->get_target_cluster().get_offset() == target_offset0) {
            //                 if (H_child->is_low_rank()) {
            //                     // auto vK = K->mat_hmat(H_child->get_low_rank_data()->Get_V());
            //                     auto vK = K->matrix_hmatrix(H_child->get_low_rank_data()->Get_V());

            //                     sr.push_back(H_child->get_low_rank_data()->Get_U());
            //                     sr.push_back(vK);

            //                     of.push_back(H_child->get_target_cluster().get_offset());
            //                     of.push_back(K->get_source_cluster().get_offset());
            //                 } else {
            //                     sh.push_back(H_child.get());
            //                     sh.push_back(K);
            //                 }
            //             }
            //         }
            //     }
            // } else if ((H_children.size() == 0) and (K_children.size() == 0)) {
            //     std::cout << " mauvaise restriction : matrice insécable a gauche et a droite" << std::endl;
            // }
        }

        // SumExpression res(sr, sh, of, target_size0, target_offset0, source_size0, source_offset0, flag);
        return res;
    }

    /////////////
    // L'ultime restict 19/06

    // const SumExpression ultim_restrict(int target_size0, int source_size0, int target_offset0, int source_offset0) {
    //     std::vector<const HMatrixType *> sh;
    //     auto of = offset;
    //     auto sr = Sr;
    //     for (int rep = 0; rep < Sh.size() / 2; ++rep) {
    //         auto &H        = Sh[2 * rep];
    //         auto &K        = Sh[2 * rep + 1];
    //         auto rho_child = H->get_target_cluster().get_children();
    //         for (auto &child : rho_child) {
    //             auto &H_child = H->get_block(target_size0, child->get_size(), target_offset0, child->offset());
    //             auto &K_child = K->get_block(child->get_size(), source_size0, child->get_offset(), source_offset0);
    //             ///////// Si on coupe trop on met juste les plus petits possibles en dense dans sr
    //             if ((H_child->get_source_cluster().get_size() != child->get_size()) and (K_child->get_target_cluster().get_size() != child->get_size())) {
    //                 H_child = H->get_block(target_size0, H->get_source_cluster().get_size(), target_offset0, H->get_source_cluster().get_offset());
    //                 K_child = K->get_block(K->get_target_cluster().get_size(), source_size0, K->get_target_cluster().get_offset(), source_offset0);
    //                 auto hr = H_child->extract(target_size0, child->get_size(), target_offset0, child->get_offset());
    //                 auto rk = K_child->extract(child->get_size(), source_size0, child->get_offset(), source_offset0);
    //                 sr.push_back(hr);
    //                 sr.push_back(rk);
    //                 of.push_back(target_offset0);
    //                 of.push_back(source_offset0);
    //             } else if ((H_child->get_source_cluster().get_size() != child->get_size())) {
    //                 H_child = H->get_block(target_size0, H->get_source_cluster().get_size(), target_offset0, H->get_source_cluster().get_offset());
    //                 auto hr = H_child->extract(target_size0, child->get_size(), target_offset0, child->get_offset());
    //                 Matrix<CoefficientPrecision> rk(child->get_size(), source_size0);
    //                 copy_to_dense(*K_child, rk.data());
    //                 sr.push_back(hr);
    //                 sr.push_back(K_child);
    //                 of.push_back(target_offset0);
    //                 of.push_back(source_offset0);
    //             } else if ((K_child->get_target_cluster().get_size() != child->get_size())) {
    //                 auto rK     = K->get_block(K->get_target_cluster().get_size(), source_size0, K->get_target_cluster().get_offset(), source_offset0);
    //                 auto rk = rK->extract(child->get_size(), source_size0, child->get_offset(), source_offset0);
    //                 Matrix<CoefficientPrecision> hr(target_size0, child->get_size());
    //                 copy_to_dense(*H_child, hr.data());
    //                 sr.push_back(hr);
    //                 sr.push_back(rk);
    //                 of.push_back(target_offset0);
    //                 of.push_back(source_offset0);
    //             } else {
    //                 if (H_child->is_low_rank() or K_child->is_low_rank()) {
    //                     if (H_child->is_low_rank() or K_child->is_low_rank()) {
    //                         if (H_child->is_low_rank() and K_child->is_low_rank()) {
    //                             auto v = H_child->get_low_rank_data()->Get_V() * K_child->get_low_rank_data()->Get_U() * K_child->get_low_rank_data()->Get_V();
    //                             sr.push_back(H_child->get_low_rank_data()->Get_U());
    //                             sr.push_back(v);
    //                             of.push_back(H_child->get_target_cluster().get_offset());
    //                             of.push_back(K_child->get_source_cluster().get_offset());
    //                         } else if ((H_child->is_low_rank()) and !(K_child->is_low_rank())) {
    //                             auto vK = K_child->mat_hmat(H_child->get_low_rank_data()->Get_V());
    //                             sr.push_back(H_child->get_low_rank_data()->Get_U());
    //                             sr.push_back(vK);
    //                             of.push_back(H_child->get_target_cluster().get_offset());
    //                             of.push_back(K_child->get_source_cluster().get_offset());
    //                         } else if (!(H_child->is_low_rank()) and (K_child->is_low_rank())) {
    //                             auto Hu = H_child->hmat_mat(K_child->get_low_rank_data()->Get_U());
    //                             sr.push_back(Hu);
    //                             sr.push_back(K_child->get_low_rank_data()->Get_V());
    //                             of.push_back(H_child->get_target_cluster().get_offset());
    //                             of.push_back(K_child->get_source_cluster().get_offset());
    //                         }
    //                     } else {
    //                         sh.push_back(H_child.get());
    //                         sh.push_back(K_child.get());
    //                     }
    //                 }
    //             }
    //         }
    //         SumExpression res(sr, sh, of, target_size0, target_offset0, source_size0, source_offset0);
    //         return res;
    //     }

    ////////////////////////////////////
    //// CA MARCHE PAS ET C EST MEME PAS PLUS RAPIDE
    const SumExpression
    ultim_restrict(int target_size0, int target_offset0, int source_size0, int source_offset0) const {
        std::vector<const HMatrixType *> sh;
        auto of = offset;
        auto sr = Sr;
        for (int rep = 0; rep < Sh.size() / 2; ++rep) {
            auto &H          = Sh[2 * rep];
            auto &K          = Sh[2 * rep + 1];
            auto &H_children = H->get_children();
            auto &K_children = K->get_children();
            bool test_k      = true;
            bool test_h      = true;
            int reference_sh = sh.size();
            int reference_sr = sr.size();
            for (int rep_h = 0; rep_h < H->get_children().size() and test_h; ++rep_h) {
                auto &H_child = H->get_children()[rep_h];
                if (H_child->get_source_cluster().get_size() == H->get_source_cluster().get_size()) {
                    test_h = false;
                } else if ((H_child->get_target_cluster().get_size() == target_size0) and (H_child->get_target_cluster().get_offset() == target_offset0)) {
                    for (int rep_k = 0; rep_k < K->get_children().size() and test_k; ++rep_k) {
                        auto &K_child = K->get_children()[rep_k];
                        if (K_child->get_target_cluster().get_size() == K->get_target_cluster().get_size()) {
                            test_k = false;
                        } else if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
                            if ((K_child->get_target_cluster().get_size() == H_child->get_source_cluster().get_size())
                                and (K_child->get_target_cluster().get_offset() == H_child->get_source_cluster().get_offset())) {
                                if (H_child->is_low_rank() or K_child->is_low_rank()) {
                                    if (H_child->is_low_rank() and K_child->is_low_rank()) {
                                        auto v = H_child->get_low_rank_data()->Get_V() * K_child->get_low_rank_data()->Get_U() * K_child->get_low_rank_data()->Get_V();
                                        sr.push_back(H_child->get_low_rank_data()->Get_U());
                                        sr.push_back(v);
                                        of.push_back(H_child->get_target_cluster().get_offset());
                                        of.push_back(K_child->get_source_cluster().get_offset());
                                    } else if ((H_child->is_low_rank()) and !(K_child->is_low_rank())) {
                                        auto vK = K_child->mat_hmat(H_child->get_low_rank_data()->Get_V());
                                        sr.push_back(H_child->get_low_rank_data()->Get_U());
                                        sr.push_back(vK);
                                        of.push_back(H_child->get_target_cluster().get_offset());
                                        of.push_back(K_child->get_source_cluster().get_offset());
                                    } else if (!(H_child->is_low_rank()) and (K_child->is_low_rank())) {
                                        auto Hu = H_child->hmat_mat(K_child->get_low_rank_data()->Get_U());
                                        sr.push_back(Hu);
                                        sr.push_back(K_child->get_low_rank_data()->Get_V());
                                        of.push_back(H_child->get_target_cluster().get_offset());
                                        of.push_back(K_child->get_source_cluster().get_offset());
                                    }
                                } else {
                                    sh.push_back(H_child.get());
                                    sh.push_back(K_child.get());
                                }
                            }
                        }
                    }
                }
            }
            if (!(test_h and test_k)) {
                // CA NE SERT A RIEN CAR CE NEST PAS POSSIBLES mais au cas ou on vérifie u'on a rien écrit
                for (int it = 0; it < (sh.size() - reference_sh); ++it) {
                    sh.pop_back();
                }
                for (int it = 0; it < (sr.size() - reference_sr); ++it) {
                    sr.pop_back();
                    of.pop_back();
                    of.pop_back();
                }
                ////////////////////////////
                /// c'est un peu bourrin mais c'est vraiment la seule chose a faire et puis ca ressemble a UV^T
                Matrix<CoefficientPrecision> hdense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                copy_to_dense(*H, hdense.data());
                Matrix<CoefficientPrecision> kdense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                copy_to_dense(*K, hdense.data());
                sr.push_back(hdense);
                sr.push_back(kdense);
                of.push_back(H->get_target_cluster().get_offset());
                of.push_back(K->get_source_cluster().get_offset());
            }
        }
        SumExpression res(sr, sh, of, target_size0, target_offset0, source_size0, source_offset0);
        return res;
    }
};

/////////////////////////////////
/// sum expression avec update
template <typename CoefficientPrecision, typename CoordinatePrecision>
class SumExpression_update : public VirtualGenerator<CoefficientPrecision> {
  private:
    using HMatrixType = HMatrix<CoefficientPrecision, CoordinatePrecision>;
    using LRtype      = LowRankMatrix<CoefficientPrecision, CoordinatePrecision>;

    // WARNING : on stocke U et V ----------------> c'est dommage on est obligé de transposé V , du coup si on était en col major ce serait super rapide
    std::vector<Matrix<CoefficientPrecision>> Sr; /// on stocke que deux matrices , pas la liste
    std::vector<const HMatrixType *> Sh;          //// sum (HK)
    int target_size;
    int target_offset;
    int source_size;
    int source_offset;
    bool restrictibe = true;

  public:
    SumExpression_update(const HMatrixType *A, const HMatrixType *B) {
        Sh.push_back(A);
        Sh.push_back(B);
        target_size   = A->get_target_cluster().get_size();
        target_offset = A->get_target_cluster().get_offset();
        source_size   = B->get_source_cluster().get_size();
        source_offset = B->get_source_cluster().get_offset();
    }

    SumExpression_update(const std::vector<Matrix<CoefficientPrecision>> &sr, const std::vector<const HMatrixType *> &sh, const int &target_size0, const int &source_size0, const int &target_offset0, const int &source_offset0, const bool flag) {
        Sr            = sr;
        Sh            = sh;
        target_size   = target_size0;
        target_offset = target_offset0;
        source_size   = source_size0;
        source_offset = source_offset0;
        restrictibe   = flag;
    }
    ////////////////////////////
    //// Getter
    const std::vector<Matrix<CoefficientPrecision>> get_sr() const { return Sr; }
    const std::vector<const HMatrixType *> get_sh() const { return Sh; }
    int get_nr() const { return target_size; }
    int get_nc() const { return source_size; }
    int get_target_offset() const { return target_offset; }
    int get_source_offset() const { return source_offset; }
    int get_target_size() const { return target_size; }
    int get_source_size() const { return source_size; }
    const bool is_restrectible() const { return restrictibe; }

    /////////////////////////
    //// Setter

    void set_sr(const std::vector<Matrix<CoefficientPrecision>> &sr) {
        Sr = sr;
    }
    void set_restr(const bool flag) { restrictibe = flag; }
    void add_HK(const HMatrixType *H, const HMatrixType *K) {
        Sh.push_back(H);
        Sh.push_back(K);
    }

    /////////////////////////////////////////////////////////////////////////
    /// PRODUCT SUMEXPR vect
    ////////////////////////////////////////////////////////////

    // prod pour sum expr avec accumulate update
    std::vector<CoefficientPrecision> new_prod(const char T, const std::vector<CoefficientPrecision> &x) const {
        if (T == 'N') {
            std::vector<CoefficientPrecision> y(target_size, 0.0);
            if (Sr.size() > 0) {
                auto u = Sr[0];
                auto v = Sr[1];
                std::vector<CoefficientPrecision> ytemp(v.nb_rows(), 0.0);
                v.add_vector_product('N', 1.0, x.data(), 0.0, ytemp.data());
                u.add_vector_product('N', 1.0, ytemp.data(), 1.0, y.data());
            }

            for (int k = 0; k < Sh.size() / 2; ++k) {
                auto H = Sh[2 * k];
                auto K = Sh[2 * k + 1];
                std::vector<CoefficientPrecision> y_temp(K->get_target_cluster().get_size(), 0.0);
                K->add_vector_product('N', 1.0, x.data(), 0.0, y_temp.data());
                H->add_vector_product('N', 1.0, y_temp.data(), 1.0, y.data());
            }
            return (y);
        }

        else {
            std::vector<CoefficientPrecision> y(source_size, 0.0);
            if (Sr.size() > 0) {
                auto u = Sr[0];
                auto v = Sr[1];
                std::vector<CoefficientPrecision> ytemp(u.nb_rows(), 0.0);
                u.add_vector_product('T', 1.0, x.data(), 0.0, ytemp.data());
                v.add_vector_product('T', 1.0, ytemp.data(), 1.0, y.data());
            }

            // y = y +(x* urestr) * vrestrx;

            for (int k = 0; k < Sh.size() / 2; ++k) {
                auto H = Sh[2 * k];
                auto K = Sh[2 * k + 1];
                std::vector<CoefficientPrecision> y_temp(H->get_source_cluster().get_size(), 0.0);
                H->add_vector_product('T', 1.0, x.data(), 1.0, y_temp.data());
                K->add_vector_product('T', 1.0, y_temp.data(), 1.0, y.data());
            }
            return (y);
        }
    }

    ////////////
    /// get _row & get_col by performing Sum_expr vector multiplication
    std::vector<CoefficientPrecision> get_col(const int &l) const {
        std::vector<double> e_l(source_size);
        e_l[l]   = 1.0;
        auto col = this->new_prod('N', e_l);
        return col;
    }
    std::vector<CoefficientPrecision> get_row(const int &k) const {
        std::vector<double> e_k(target_size);
        e_k[k]   = 1.0;
        auto row = this->new_prod('T', e_k);
        return row;
    }

    // do Sr = Sr +AB with QR svd
    // U1V1^T+U2V2^T = (U1, U2)(V1,V2)^T = (Q1R1)(Q2R2)T =
    void actualise(const Matrix<CoefficientPrecision> &A, const Matrix<CoefficientPrecision> &B, const int &rk, const CoefficientPrecision &epsilon, bool flag) {
        // first call
        if (Sr.size() == 0) {
            Sr.push_back(A);
            Sr.push_back(B);
        }
        // update
        else {
            // on regarde si la somme des rang est pas plus grand que leading dimension ( ca peut arriver sur les petits blocs admissibles)
            auto U = Sr[0];
            auto V = Sr[1];
            // on concatène
            const int N = U.nb_cols() + A.nb_cols(); // rank1+rank2
            if (U.nb_rows() >= N && V.nb_cols() >= N) {
                Matrix<CoefficientPrecision> big_U(U.nb_rows(), N);
                Matrix<CoefficientPrecision> big_VT(B.nb_cols(), N);

                // big_U c'est facile
                // for (int j = 0; j < U.nb_cols(); ++j) {
                //     std::copy(U.data() + j * U.nb_rows(), U.data() + (j + 1) * U.nb_rows(), big_U.data() + j * U.nb_rows());
                // }
                // for (int j = 0; j < A.nb_cols(); ++j) {
                //     std::copy(A.data() + j * U.nb_rows(), A.data() + (j + 1) * U.nb_rows(), big_U.data() + (j + U.nb_cols()) * U.nb_rows());
                // }
                for (int k = 0; k < U.nb_rows(); ++k) {
                    for (int l = 0; l < U.nb_cols(); ++l) {
                        big_U(k, l) = U(k, l);
                    }
                    for (int l = 0; l < A.nb_cols(); ++l) {
                        big_U(k, l + U.nb_cols()) = A(k, l);
                    }
                }
                // big_V
                /// concatenation + transposition of bigVT -> bigV     (oui c'est chiant il y a des ^T partout)
                for (int i = 0; i < V.nb_rows(); ++i) {
                    std::vector<CoefficientPrecision> ei(V.nb_rows(), 0);
                    ei[i] = 1.0;
                    std::vector<CoefficientPrecision> rowtemp(V.nb_cols());
                    V.add_vector_product('T', 1.0, ei.data(), 1.0, rowtemp.data());
                    std::copy(rowtemp.begin(), rowtemp.begin() + V.nb_cols(), big_VT.data() + i * V.nb_cols());
                }
                for (int i = 0; i < B.nb_rows(); ++i) {
                    std::vector<CoefficientPrecision> ei(B.nb_rows(), 0);
                    ei[i] = 1.0;
                    std::vector<CoefficientPrecision> rowtemp(V.nb_cols()); // oui V , ca doit être la même taille
                    B.add_vector_product('T', 1.0, ei.data(), 1.0, rowtemp.data());
                    std::copy(rowtemp.begin(), rowtemp.begin() + V.nb_cols(), big_VT.data() + (i + V.nb_rows()) * V.nb_cols());
                }
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

                Blas<CoefficientPrecision>::gemm("N", "T", &N, &N, &N, &alpha_0, Ru.data(), &N, Rv.data(), &N, &beta_0, R_uR_v.data(), &N);
                // truncated SVD sur R_uR_v
                std::vector<CoefficientPrecision> S(N, 0.0);
                Matrix<CoefficientPrecision> Uu(N, N);
                Matrix<CoefficientPrecision> VT(N, N);
                int info;
                int lwork = 10 * N;
                std::vector<double> rwork(lwork);
                std::vector<CoefficientPrecision> work(lwork);
                auto tempp = R_uR_v;
                Lapack<CoefficientPrecision>::gesvd("A", "A", &N, &N, R_uR_v.data(), &N, S.data(), Uu.data(), &N, VT.data(), &N, work.data(), &lwork, rwork.data(), &info);
                Matrix<CoefficientPrecision> SS(N, N);
                for (int k = 0; k < N; ++k) {
                    SS(k, k) = S[k];
                }
                // std::cout << "erreur svd : " << normFrob(Uu * SS * VT - tempp) / normFrob(tempp) << std::endl;
                // truncation index
                int trunc = 0;
                while (S[trunc] > epsilon) {
                    trunc += 1;
                }
                // if (rk == -1) {
                //     while (S[trunc] > epsilon) {
                //         trunc += 1;
                //     }
                // } else if (epsilon == -1) {
                //     trunc = rk;
                // } else {
                //     std::cout << "il faut un critère rank ou epsilon , pas encore implémenter rk && epsilon" << std::endl;
                // }
                // on tronque U S et V
                // ca on pourrait le faire en décallant de N tout les coefficients de S avec un resize special
                if (trunc < N / 2) {
                    Matrix<CoefficientPrecision> Strunc(trunc, trunc);
                    for (int k = 0; k < trunc; ++k) {
                        Strunc(k, k) = S[k];
                    }
                    ///// truncation of svd vectors
                    Matrix<CoefficientPrecision> Utrunc(N, trunc);
                    Matrix<CoefficientPrecision> Vtrunc(trunc, N);
                    for (int k = 0; k < N; ++k) {
                        for (int l = 0; l < trunc; ++l) {
                            Utrunc(k, l) = Uu(k, l);
                        }
                    }
                    // std::cout << "VT " << VT.nb_rows() << ',' << VT.nb_cols() << "    Vtrunc " << Vtrunc.nb_rows() << ',' << Vtrunc.nb_cols() << std::endl;
                    for (int k = 0; k < trunc; ++k) {
                        for (int l = 0; l < N; ++l) {
                            Vtrunc(k, l) = VT(k, l);
                        }
                    }
                    ///// fast trunc
                    // auto Utrunc = trunc_row(&Uu, trunc);
                    // auto Vtrunc = trunc_col(&VT, trunc);
                    // U = big_U Utrunc Strunc     --------------   V= VT big_V ;
                    int mm     = big_U.nb_rows();
                    int nn     = big_VT.nb_rows();
                    int truncc = trunc;
                    // std::cout << "norme avant trunc :" << normFrob(Uu) << ',' << normFrob(VT) << std::endl;
                    // std::cout << "norme aprés trunc " << normFrob(Utrunc) << ',' << normFrob(Vtrunc) << std::endl;
                    // Matrix<CoefficientPrecision> temp(U.nb_rows(), trunc);
                    // Matrix<CoefficientPrecision> res_U(U.nb_rows(), trunc);
                    // Matrix<CoefficientPrecision> res_V(trunc, big_VT.nb_rows());
                    // big_U.add_matrix_product('N', 1.0, Utrunc.data(), 1.0, temp.data(), trunc);
                    // temp.add_matrix_product('N', 1.0, Strunc.data(), 0.0, res_U.data(), trunc);
                    // (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

                    // Blas<CoefficientPrecision>::gemm("N", "T", &trunc, &source_size, &trunc, &alpha_0, Vtrunc.data(), &trunc, big_VT.data(), &source_size, &beta_0, res_V.data(), &source_size);
                    auto res_V = Vtrunc * Qv.transp(Qv);
                    auto res_U = Qu * Utrunc * Strunc;
                    // Finally
                    Sr[0] = res_U;
                    Sr[1] = res_V;
                    // std::cout << "actualise a push U et V de norme :" << normFrob(res_U) << "   ,    " << normFrob(res_V) << std::endl;
                    // std::cout << "actualise a mis des matrices de normes " << normFrob(res_U) << ',' << normFrob(res_V) << std::endl;
                } else {
                    // std::cout << "pas de bonne approximation de UV+AB" << std::endl;
                    flag = false;
                }
            } else {
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
                    int trunc = 0;
                    while (S[trunc] > epsilon) {
                        trunc += 1;
                    }
                    if (trunc < Nn / 2) {
                        Matrix<CoefficientPrecision> Strunc(trunc, trunc);
                        for (int k = 0; k < trunc; ++k) {
                            Strunc(k, k) = S[k];
                        }
                        ///// truncation of svd vectors
                        Matrix<CoefficientPrecision> Utrunc(nr, trunc);
                        Matrix<CoefficientPrecision> Vtrunc(trunc, nc);
                        for (int k = 0; k < nr; ++k) {
                            for (int l = 0; l < trunc; ++l) {
                                Utrunc(k, l) = Uu(k, l);
                            }
                        }
                        // std::cout << "VT " << VT.nb_rows() << ',' << VT.nb_cols() << "    Vtrunc " << Vtrunc.nb_rows() << ',' << Vtrunc.nb_cols() << std::endl;
                        for (int k = 0; k < trunc; ++k) {
                            for (int l = 0; l < nc; ++l) {
                                Vtrunc(k, l) = VT(k, l);
                            }
                        }
                        auto res = Utrunc * Strunc;
                        Sr[0]    = res;
                        Sr[1]    = Vtrunc;
                        // std::cout << "                actualise a push U et V de norme :" << normFrob(Sr[0]) << "   ,    " << normFrob(Sr[1]) << std::endl;
                    } else {
                        std::cout << "pas de bonne approximation de UV +AB" << std::endl;
                        flag = false;
                    }
                } else {
                    std::cout << "nc> nr" << std::endl;
                    flag = false;
                }
            }
        }
    }

    std::vector<Matrix<CoefficientPrecision>> compute_lr_update(const Matrix<CoefficientPrecision> &A, const Matrix<CoefficientPrecision> &B, const int &rk, const CoefficientPrecision &epsilon) {
        std::vector<Matrix<CoefficientPrecision>> res;
        // first call
        if (Sr.size() == 0) {
            res.push_back(A);
            res.push_back(B);
            std::cout << "----------------------on push U et V de taille " << A.nb_rows() << ',' << A.nb_cols() << ',' << B.nb_rows() << ',' << B.nb_cols() << std::endl;
        }
        // update
        else {
            // on regarde si la somme des rang est pas plus grand que leading dimension ( ca peut arriver sur les petits blocs admissibles)
            auto U = Sr[0];
            auto V = Sr[1];
            std::cout << "appel update avec " << U.nb_rows() << ',' << U.nb_cols() << "+" << A.nb_rows() << ',' << A.nb_cols() << "|" << V.nb_rows() << ',' << V.nb_cols() << '+' << B.nb_rows() << ',' << B.nb_cols() << std::endl;
            const int N = U.nb_cols() + A.nb_cols(); // rank1+rank2
            // on regarde si c'est interressant de concaténer
            if (U.nb_rows() > N && V.nb_cols() > N) {
                // on concatène
                auto big_VT = conc_row_T(V, B);

                auto big_U = conc_col(U, A);
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
                Blas<CoefficientPrecision>::gemm("N", "T", &N, &N, &N, &alpha_0, Ru.data(), &N, Rv.data(), &N, &beta_0, R_uR_v.data(), &N); // j'ai besoin du "transb" du coup je prend gemm
                // truncated SVD sur R_uR_v
                Matrix<CoefficientPrecision> Uu(N, N);
                Matrix<CoefficientPrecision> VT(N, N);
                auto S = compute_svd(R_uR_v, Uu, VT);
                // Matyrix<CoefficientPrecision> SS(N, N);
                // for (int k = 0; k < N; ++k) {
                //     SS(k, k) = S[k];
                // }
                // std::cout << "erreur svd : " << normFrob(Uu * SS * VT - tempp) / normFrob(tempp) << std::endl;
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
                    Blas<CoefficientPrecision>::gemm("N", "N", &N, &trunc, &trunc, &alpha_0, Utrunc.data(), &N, Strunc.data(), &trunc, &beta_0, Us.data(), &N);
                    Matrix<CoefficientPrecision> res_U(target_size, trunc);
                    Blas<CoefficientPrecision>::gemm("N", "N", &target_size, &trunc, &N, &alpha_0, Qu.data(), &target_size, Us.data(), &N, &beta_0, Us.data(), &target_size);

                    // Finally
                    res.push_back(res_U);
                    res.push_back(res_V);
                    std::cout << "on push U et V de taille " << res_U.nb_rows() << ',' << res_U.nb_cols() << ',' << res_V.nb_rows() << ',' << res_V.nb_cols() << std::endl;

                } else {
                    // std::cout << "pas de bonne approximation de UV+AB" << std::endl;
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
                        Blas<CoefficientPrecision>::gemm("N", "N", &nr, &nc, &trunc, &alpha_0, Utrunc.data(), &nr, Strunc.data(), &trunc, &beta_0, Us.data(), &std::max(nr, nc)); // j'ai besoin du "transb" du coup je prend gemm
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
        }
        return res;
    }

    ///////////////////////////////////////////////////////////////////
    /// Copy sub matrix ----------------> sumexpr = virtual_generator
    ////////////////////////////////////////////////////

    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override { // on regarde si il faut pas juste une ligne ou une collonne
        Matrix<CoefficientPrecision> mat(M, N);
        // mat.assign(M, N, ptr, false);
        std::cout << "mat :" << normFrob(mat) << std::endl;
        int ref_row = row_offset - target_offset;
        int ref_col = col_offset - source_offset;
        if (M == 1) {
            auto row = this->get_row(ref_row);
            if (N == source_size) {
                std::copy(row.begin(), row.begin() + N, ptr);

            } else {
                for (int l = 0; l < N; ++l) {
                    ptr[l] = row[l + ref_col];
                }
            }
        } else if (N == 1) {
            auto col = this->get_col(ref_col);
            if (M == target_size) {
                std::copy(col.begin(), col.begin() + M, ptr);
                // ptr = col.data();
                // std::cout << normFrob(mat) << std::endl;

            } else {
                for (int k = 0; k < M; ++k) {
                    ptr[k] = col[k + ref_row];
                }
            }
        }

        else {
            // donc la on veut un bloc, on va faire un choix arbitraire pour appeller copy_to_dense
            double alpha_0 = 1.0;

            if ((M + N) > (target_size + source_size) / 2) {
                // std::cout << "énorme bloc densifié : " << M << ',' << N << " dans un bloc de taille :  " << target_size << ',' << source_size << "|" << target_offset << ',' << source_offset << std::endl;

                for (int k = 0; k < Sh.size() / 2; ++k) {
                    auto &H = Sh[2 * k];
                    auto &K = Sh[2 * k + 1];
                    Matrix<CoefficientPrecision> hdense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                    Matrix<CoefficientPrecision> kdense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                    copy_to_dense(*H, hdense.data());
                    std::cout << "1" << std::endl;
                    copy_to_dense(*K, kdense.data());
                    std::cout << "2" << std::endl;
                    int r       = H->get_source_cluster().get_size();
                    auto hrestr = hdense.get_block(M, r, ref_row, 0);
                    auto krestr = kdense.get_block(r, N, 0, ref_col);
                    std::cout << normFrob(hrestr) << ',' << normFrob(krestr) << ',' << normFrob(hrestr - krestr) << std::endl;
                    Blas<CoefficientPrecision>::gemm("N", "N", &M, &N, &r, &alpha_0, hrestr.data(), &std::max(M, r), krestr.data(), &std::max(N, r), &alpha_0, ptr, &std::max(M, N));
                }
                for (int k = 0; k < Sr.size() / 2; ++k) {
                    auto U       = Sr[2 * k];
                    auto V       = Sr[2 * k + 1];
                    auto U_restr = U.get_block(M, U.nb_cols(), row_offset - target_offset, 0);
                    auto V_restr = V.get_block(V.nb_rows(), N, 0, col_offset - source_offset);
                    int rk       = U_restr.nb_cols();
                    Matrix<CoefficientPrecision> prod(M, N);
                    Blas<CoefficientPrecision>::gemm("N", "N", &M, &N, &rk, &alpha_0, U_restr.data(), &std::max(M, rk), V_restr.data(), &std::max(N, rk), &alpha_0, ptr, &std::max(M, N));
                }

            } else {
                for (int k = 0; k < M; ++k) {
                    auto row = this->get_row(ref_row + k);
                    for (int l = 0; l < N; ++l) {
                        ptr[k + M * l] = row[l + ref_col];
                    }
                }
            }
        }
    }

    // ///////////////////////////////////////////////////////
    // // RESTRICT
    // /////////////////////////////////////////////

    const SumExpression_update Restrict(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) const {
        int target_size0   = t.get_size();
        int target_offset0 = t.get_offset();
        int source_size0   = s.get_size();
        int source_offset0 = s.get_offset();

        std::vector<const HMatrixType *> sh;
        // on recopie UV
        std::vector<Matrix<CoefficientPrecision>> sr;
        bool flag = true;
        if (Sr.size() > 0) {
            // on restreint
            auto U      = Sr[0];
            auto V      = Sr[1];
            auto urestr = U.get_block(target_size0, U.nb_cols(), target_offset0 - target_offset, 0.0);
            auto vrestr = V.get_block(U.nb_cols(), source_size0, 0.0, source_offset0 - source_offset);
            sr.push_back(urestr);
            sr.push_back(vrestr);
        }
        SumExpression_update res(sr, sh, target_size0, source_size0, target_offset0, source_offset0, flag);
        for (int rep = 0; rep < Sh.size() / 2; ++rep) {
            auto &H          = Sh[2 * rep];
            auto &K          = Sh[2 * rep + 1];
            auto &H_children = H->get_children();
            auto &K_children = K->get_children();
            if ((H_children.size() > 0) and (K_children.size() > 0)) {
                for (auto &H_child : H_children) {
                    if ((H_child->get_target_cluster().get_size() == target_size0) and (H_child->get_target_cluster().get_offset() == target_offset0)) {
                        for (auto &K_child : K_children) {
                            if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
                                if ((K_child->get_target_cluster().get_size() == H_child->get_source_cluster().get_size())
                                    and (K_child->get_target_cluster().get_offset() == H_child->get_source_cluster().get_offset())) {
                                    if (H_child->is_low_rank() or K_child->is_low_rank()) {
                                        if (H_child->is_low_rank() and K_child->is_low_rank()) {
                                            int M = H_child->get_low_rank_data()->Get_V().nb_rows();
                                            int N = K_child->get_low_rank_data()->Get_U().nb_cols();
                                            int k = H_child->get_low_rank_data()->Get_V().nb_cols();

                                            int vk        = K_child->get_low_rank_data()->Get_U().nb_cols();
                                            int Nv        = K_child->get_source_cluster().get_size();
                                            double alpha0 = 1.0;
                                            double beta0  = 0.0;
                                            // Matrix<CoefficientPrecision> HvKu(M, k);
                                            // Matrix<CoefficientPrecision> V(k, K_child->get_source_cluster().get_size());
                                            // Blas<CoefficientPrecision>::gemm("N", "N", &M, &N, &k, &alpha0, H_child->get_low_rank_data()->Get_V().data(), &std::max(M, k), K_child->get_low_rank_data()->Get_U().data(), &std::max(N, k), &beta0, HvKu.data(), &std::max(M, N));
                                            // Blas<CoefficientPrecision>::gemm("N", "N", &M, &Nv, &vk, &alpha0, HvKu.data(), &std::max(M, vk), K_child->get_low_rank_data()->Get_V().data(), &std::max(Nv, vk), &beta0, V.data(), &std::max(M, Nv));
                                            auto V  = H_child->get_low_rank_data()->Get_V() * K_child->get_low_rank_data()->Get_U() * K_child->get_low_rank_data()->Get_V();
                                            auto uv = res.compute_lr_update(H_child->get_low_rank_data()->Get_U(), V, -1, H->get_epsilon());
                                            if (uv.size() == 0) {
                                                res.add_HK(H_child.get(), K_child.get());
                                                flag = false;
                                            } else {
                                                std::cout << " 1: on va push " << uv[0].nb_rows() << ',' << uv[0].nb_cols() << ',' << uv[1].nb_rows() << ',' << uv[1].nb_cols() << std::endl;
                                                res.set_sr(uv);
                                            }
                                        } else if ((H_child->is_low_rank()) and !(K_child->is_low_rank())) {
                                            auto vK = K_child->mat_hmat(H_child->get_low_rank_data()->Get_V());
                                            auto uv = res.compute_lr_update(H_child->get_low_rank_data()->Get_U(), vK, -1, H->get_epsilon());
                                            if (uv.size() == 0) {
                                                res.add_HK(H_child.get(), K_child.get());
                                                flag = false;
                                            } else {
                                                std::cout << " 2: on va push " << uv[0].nb_rows() << ',' << uv[0].nb_cols() << ',' << uv[1].nb_rows() << ',' << uv[1].nb_cols() << std::endl;

                                                res.set_sr(uv);
                                            }
                                        } else if (!(H_child->is_low_rank()) and (K_child->is_low_rank())) {
                                            auto Hu = H_child->hmat_mat(K_child->get_low_rank_data()->Get_U());
                                            auto uv = res.compute_lr_update(Hu, K_child->get_low_rank_data()->Get_V(), -1, H->get_epsilon());
                                            if (uv.size() == 0) {
                                                res.add_HK(H_child.get(), K_child.get());
                                                flag = false;
                                            } else {
                                                std::cout << " 3: on va push " << uv[0].nb_rows() << ',' << uv[0].nb_cols() << ',' << uv[1].nb_rows() << ',' << uv[1].nb_cols() << std::endl;

                                                res.set_sr(uv);
                                            }
                                        }
                                    } else {
                                        res.add_HK(H_child.get(), K_child.get());
                                        if ((H_child->get_children().size() == 0) && (K_child->get_children().size() == 0)) {
                                            flag = false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                std::cout << "cas pas possible ->  soit tout les fils soit aucun" << std::endl;
            }
        }
        res.set_restr(flag);
        return res;
    }
};
} // namespace htool

#endif
