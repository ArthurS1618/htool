#ifndef SUM_EXPRESSIONS_HPP
#define SUM_EXPRESSIONS_HPP
#include "../basic_types/vector.hpp"
#include <random>

namespace htool {

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
    std::vector<int> offset; // target de u et source de v -> 2*size Sr
    std::vector<const HMatrixType *> Sh;
    int target_size;
    int target_offset;
    int source_size;
    int source_offset;

  public:
    SumExpression(const HMatrixType *A, const HMatrixType *B) {
        Sh.push_back(A);
        Sh.push_back(B);
        target_size   = A->get_target_cluster().get_size();
        target_offset = A->get_target_cluster().get_offset();
        source_size   = B->get_source_cluster().get_size();
        source_offset = B->get_source_cluster().get_offset();
    }

    SumExpression(const std::vector<Matrix<CoefficientPrecision>> &sr, const std::vector<const HMatrixType *> &sh, const std::vector<int> &offset0, const int &target_size0, const int &target_offset0, const int &source_size0, const int &source_offset0) {
        Sr            = sr;
        Sh            = sh;
        offset        = offset0;
        target_size   = target_size0;
        target_offset = target_offset0;
        source_size   = source_size0;
        source_offset = source_offset0;
    }

    const std::vector<Matrix<CoefficientPrecision>> get_sr() const { return Sr; }
    const std::vector<const HMatrixType *> get_sh() const { return Sh; }
    int get_nr() const { return target_size; }
    int get_nc() const { return source_size; }
    int get_target_offset() const { return target_offset; }
    int get_source_offset() const { return source_offset; }
    const std::vector<int> get_offset() const { return offset; }

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
            vrestr.add_vector_product('N', 1, x.data(), 0, ytemp.data());
            urestr.add_vector_product('N', 1, ytemp.data(), 1, y.data());
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
    const std::vector<CoefficientPrecision> new_prod(const std::vector<CoefficientPrecision> &x) const {
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
            const HMatrixType &H = *Sh[2 * k];
            const HMatrixType &K = *Sh[2 * k + 1];
            std::vector<CoefficientPrecision> y_temp(K.get_target_cluster().get_size(), 0.0);
            K.add_vector_product('N', 1.0, x.data(), 0.0, y_temp.data());
            H.add_vector_product('N', 1.0, y_temp.data(), 1.0, y.data());
        }
        return (y);
    }
    /// ///////////////////////////////////////////////
    /////// get_coeff pour copy sub_matrix
    /////////////////////////////////////////
    const CoefficientPrecision get_coeff(const int &i, const int &j) const {
        std::vector<CoefficientPrecision> xj(source_size, 0.0);
        xj[j]                               = 1.0;
        std::vector<CoefficientPrecision> y = this->new_prod(xj);
        return y[i];
    }

    ///////////////////////////////////////////////////////////////////
    /// Copy sub matrix ----------------> sumexpr = virtual_generator
    ////////////////////////////////////////////////////

    ////////////////////
    /// Arthur TO DO : distinction cas parce que c'est violent quand même le copy_to_dense

    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        Matrix<CoefficientPrecision> mat(M, N);
        mat.assign(M, N, ptr, false);
        for (int k = 0; k < Sh.size() / 2; ++k) {
            auto &H = Sh[2 * k];
            auto &K = Sh[2 * k + 1];
            Matrix<CoefficientPrecision> hdense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
            Matrix<CoefficientPrecision> kdense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
            copy_to_dense(*H, hdense.data());
            copy_to_dense(*K, kdense.data());

            Matrix<CoefficientPrecision> hk = hdense * kdense;
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    mat(i, j) += hk(i + row_offset - target_offset, j + col_offset - source_offset);
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
    }

    const std::vector<int> is_restrictible() const {
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

    /////////////////////////
    //// restrcit "clean"
    //// TEST PASSED : YES -> erreur relative hmat hmat = e-5 , 73% de rang faible
    const SumExpression
    Restrict_clean(int target_size0, int target_offset0, int source_size0, int source_offset0) const {

        std::vector<const HMatrixType *> sh;
        auto of = offset;
        auto sr = Sr;
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
            } else if (H_children.size() == 0 and K_children.size() > 0) {
                if (target_size0 < H->get_target_cluster().get_size()) {
                    std::cout << " mauvais restrict , bloc dense insécablble a gauche" << std::endl;
                } else {
                    for (int l = 0; l < K_children.size(); ++l) {
                        auto &K_child = K_children[l];
                        if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
                            if (K_child->is_low_rank()) {
                                auto Hu = H->hmat_mat(K_child->get_low_rank_data()->Get_U());
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
                                auto vK = K->mat_hmat(H_child->get_low_rank_data()->Get_V());
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
        SumExpression res(sr, sh, of, target_size0, target_offset0, source_size0, source_offset0);
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
    const SumExpression ultim_restrict(int target_size0, int target_offset0, int source_size0, int source_offset0) const {
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
} // namespace htool

#endif
