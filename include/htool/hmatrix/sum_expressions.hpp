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
    DenseGenerator(const Matrix<CoefficientPrecision> &mat0) : mat(mat0){};

    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = mat(i + row_offset, j + col_offset);
            }
        }
    };
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
        // for (auto &lr : sr) {
        //     Sr.push_back(lr);
        // }
        // for (auto &h : sh) {
        //     Sh.push_back(h);
        // }
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
    const int get_nr() const { return target_size; }
    const int get_nc() const { return source_size; }
    const int get_target_offset() const { return target_offset; }
    const int get_source_offset() const { return source_offset; }
    const std::vector<int> get_offset() const { return offset; }

    const std::vector<CoefficientPrecision> prod(const std::vector<CoefficientPrecision> &x) const {
        // std::cout << "prod begin " << std::endl;
        // std::cout << x.size() << ',' << target_size << std::endl;
        std::vector<CoefficientPrecision> y(target_size, 0.0);
        for (int k = 0; k < Sr.size() / 2; ++k) {
            const Matrix<CoefficientPrecision> U = Sr[2 * k];
            const Matrix<CoefficientPrecision> V = Sr[2 * k + 1];
            int oft                              = offset[2 * k];
            int ofs                              = offset[2 * k + 1];
            // auto uv                              = U * V;
            Matrix<CoefficientPrecision> urestr;
            Matrix<CoefficientPrecision> vrestr;
            CoefficientPrecision *ptru = new CoefficientPrecision[target_size, U.nb_cols()];
            CoefficientPrecision *ptrv = new CoefficientPrecision[V.nb_rows(), source_size];

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
            // Matrix<CoefficientPrecision> uvrestr(target_size, source_size);
            // uvrestr.assign(target_size, source_size, ptr, true);
            // Matrix<CoefficientPrecision> uvrestr(target_size, source_size);
            // for (int i = 0; i < target_size; ++i) {
            //     for (int j = 0; j < source_size; ++j) {
            //         uvrestr(i, j) = uv(i + target_offset - oft, j + source_offset - ofs);
            //     }
            // }
            y = y + urestr * vrestr * x;
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
        // std::cout << "prod end " << std::endl;
        return (y);
    }

    /////////////////////////////////////////////////////////////////////////
    /// NEW PROD
    /// un peu plus rapide que prd
    ////////////////////////////////////////////////////////////
    const std::vector<CoefficientPrecision> new_prod(const std::vector<CoefficientPrecision> &x) const {
        // std::cout << "prod begin " << std::endl;
        // std::cout << x.size() << ',' << target_size << std::endl;
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

    // void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
    //     for (int k = 0; k < M; ++k) {
    //         for (int l = 0; l < N; ++l) {
    //             ptr[k + M * l] = this->get_coeff(k + row_offset - target_offset, l + col_offset - source_offset);
    //         }
    //     }
    // }

    // void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
    //     for (int l = 0; l < N; ++l) {
    //         std::vector<CoefficientPrecision> x(N, 0.0);
    //         x[l]       = 1;
    //         auto col_l = this->new_prod(x);
    //         for (int k = 0; k < M; ++k) {
    //             ptr[k + l * M] = col_l[k];
    //         }
    //     }
    // }

    ////////////
    /// lui marche trés bien mais j'aimerais voire ce sue ca change de faire avec le add_matrix_product

    ///

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

    // void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
    //     Matrix<CoefficientPrecision> mat(this->get_nr(), this->get_nc());
    //     std::cout << "ca commence " << std ::endl;
    //     for (int k = 0; k < Sh.size() / 2; ++k) {
    //         auto &H = Sh[2 * k];
    //         auto &K = Sh[2 * k + 1];
    //         Matrix<CoefficientPrecision> kdense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
    //         copy_to_dense(*K, kdense.data());
    //         std::vector<CoefficientPrecision> k_major(kdense.nb_rows() * kdense.nb_cols());

    //         for (int i = 0; i < kdense.nb_rows(); ++i) {
    //             for (int j = 0; j < kdense.nb_cols(); ++j) {
    //                 k_major[i * kdense.nb_cols() + j] = kdense.data()[i + j * kdense.nb_rows()];
    //             }
    //         }

    //         H->add_matrix_product_row_major('N', 1.0, k_major.data(), 1.0, mat.data(), kdense.nb_cols());
    //     }
    //     std::cout << "ca finit" << std::endl;
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
    //         std::vector<CoefficientPrecision> v_major(V.nb_rows() * source_size);
    //         for (int i = 0; i < V_restr.nb_rows(); ++i) {
    //             for (int j = 0; j < source_size; ++j) {
    //                 v_major[i * source_size + j] = V_restr.data()[i + j * V_restr.nb_rows()];
    //             }
    //         }
    //         U_restr.add_matrix_product('N', 1.0, v_major.data(), 1.0, mat.data(), mat.nb_cols());
    //     }
    //     Matrix<CoefficientPrecision> mat_unmajor(target_size, source_size);
    //     for (int k = 0; k < target_size; ++k) {
    //         for (int l = 0; l < source_size; ++l) {
    //             mat_unmajor.data()[l * target_size + k] = mat.data()[k * source_size + l];
    //         }
    //     }
    //     std::cout << "norme frob : " << normFrob(mat_unmajor) << std::endl;
    //     std::copy(mat.data(), mat.data() + target_size * source_size, ptr);
    // }

    // on multiplie a gauche par (tau , root_target) * Sumexpr * ( source , sigma ) , ou tu met des 1 sur les diagonales
    // void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
    //     Matrix<CoefficientPrecision> Id_tau(M, this->get_nr());
    //     Matrix<CoefficientPrecision> Id_sigma(this->get_nc(), N);
    //     Matrix<CoefficientPrecision> res(M, N);
    //     for (int k = 0; k < M; ++k) {
    //         Id_tau(k, k + row_offset) = 1;
    //     }
    //     for (int k = 0; k < N, ; ++k) {
    //         Id_sigma(k + col_offset, k) = 1;
    //     }
    //     for (int k = 0; k < sr.size(); ++k) {
    //         auto U = Sr[2 * k];
    //         auto V = Sr[2 * k + 1];
    //     }
    // }

    // const std::vector<int> is_restrictible() const {
    //     std::vector<int> test(2, 0.0);
    //     int k      = 0;
    //     bool test1 = true;
    //     bool test2 = true;

    //     while (k < Sh.size() / 2 and test1) {
    //         auto H = Sh[2 * k];
    //         auto K = Sh[2 * k + 1];
    //         if (H->get_children().size() == 0) {
    //             test[0] = 1;
    //             test1   = false;
    //         }
    //         k += 1;
    //     }
    //     k = 0;
    //     while (k < Sh.size() / 2 and test2) {
    //         auto H = Sh[2 * k];
    //         auto K = Sh[2 * k + 1];
    //         if (K->get_children().size() == 0) {
    //             test[1] = 1;
    //             test2   = false;
    //         }
    //         k += 1;
    //     }
    //     return test;
    // }
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

    // On est censé appeler restrict (tau , sigma) seulementsi il y a pas de dense plus grande que tau a gauche ou de dense plus grande que sigma a droite
    // const SumExpression
    // Restrict(int target_size0, int target_offset0, int source_size0, int source_offset0) const {
    //     std::vector<const HMatrixType *> sh;
    //     // std::vector<Matrix<CoefficientPrecision>> sr;
    //     auto of = offset;
    //     auto sr = Sr;
    //     // std::cout << "au debut" << std::endl;
    //     // if (Sr.size() > 0) {
    //     //     std::cout << Sr[0].nb_rows() << ',' << Sr[0].nb_cols() << ',' << Sr[1].nb_rows() << ',' << Sr[1].nb_cols() << std::endl;
    //     // }
    //     // std::cout << "? " << std::endl;
    //     for (int rep = 0; rep < Sh.size() / 2; ++rep) {
    //         // std::cout << "restrict sh :" << rep << '/' << Sh.size() / 2 << std::endl;
    //         auto &H          = Sh[2 * rep];
    //         auto &K          = Sh[2 * rep + 1];
    //         auto &H_children = H->get_children();
    //         auto &K_children = K->get_children();

    //         if ((H_children.size() > 0) and (K_children.size() > 0)) {
    //             for (auto &H_child : H_children) {
    //                 // int target_size_Hchild = H_child->get_target_cluster().get_size();
    //                 // int target_offset_Hchild = H->get_target_cluster().get_offset();
    //                 // int source_size_Hchild = H_child->get_source_cluster().get_size();
    //                 // int source_offset_Hchild = H->get_source_cluster().get_offset();
    //                 if ((H_child->get_target_cluster().get_size() == target_size0) and (H_child->get_target_cluster().get_offset() == target_offset0)) {
    //                     for (auto &K_child : K_children) {
    //                         if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
    //                             if ((K_child->get_target_cluster().get_size() == H_child->get_source_cluster().get_size())
    //                                 and (K_child->get_target_cluster().get_offset() == H_child->get_source_cluster().get_offset())) {
    //                                 if (H_child->is_low_rank() or K_child->is_low_rank()) {
    //                                     if (H_child->is_low_rank() and K_child->is_low_rank()) {
    //                                         Matrix<CoefficientPrecision> uh = H_child->get_low_rank_data()->Get_U();
    //                                         Matrix<CoefficientPrecision> vh = H_child->get_low_rank_data()->Get_V();
    //                                         Matrix<CoefficientPrecision> uk = K_child->get_low_rank_data()->Get_U();
    //                                         Matrix<CoefficientPrecision> vk = K_child->get_low_rank_data()->Get_V();
    //                                         Matrix<CoefficientPrecision> v  = vh * uk * vk;
    //                                         sr.push_back(uh);
    //                                         sr.push_back(v);
    //                                         of.push_back(H_child->get_target_cluster().get_offset());
    //                                         of.push_back(K_child->get_source_cluster().get_offset());
    //                                     } else if ((H_child->is_low_rank()) and !(K_child->is_low_rank())) {
    //                                         Matrix<CoefficientPrecision> u  = H_child->get_low_rank_data()->Get_U();
    //                                         Matrix<CoefficientPrecision> v  = H_child->get_low_rank_data()->Get_V();
    //                                         Matrix<CoefficientPrecision> vk = K_child->lr_hmat(v);
    //                                         sr.push_back(u);
    //                                         sr.push_back(vk);
    //                                         of.push_back(H_child->get_target_cluster().get_offset());
    //                                         of.push_back(K_child->get_source_cluster().get_offset());
    //                                     } else if (!(H_child->is_low_rank()) and (K_child->is_low_rank())) {
    //                                         // std::cout << "!3" << std::endl;
    //                                         Matrix<CoefficientPrecision> u  = K_child->get_low_rank_data()->Get_U();
    //                                         Matrix<CoefficientPrecision> v  = K_child->get_low_rank_data()->Get_V();
    //                                         Matrix<CoefficientPrecision> hu = H_child->hmat_lr(u);
    //                                         sr.push_back(hu);
    //                                         sr.push_back(v);
    //                                         of.push_back(H_child->get_target_cluster().get_offset());
    //                                         of.push_back(K_child->get_source_cluster().get_offset());
    //                                     }
    //                                 } else {
    //                                     sh.push_back(H_child.get());
    //                                     sh.push_back(K_child.get());
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         } else if (H_children.size() == 0 and K_children.size() > 0) {
    //             // H is dense otherwise HK would be in SR
    //             if (target_size0 < H->get_target_cluster().get_size()) {
    //                 std::cout << " mauvais restrict , bloc dense insécablble a gauche" << std::endl;

    //             } else {
    //                 // donc la normalement K a juste 2 fils et on prend celui qui a le bon sigma
    //                 for (int l = 0; l < K_children.size(); ++l) {
    //                     auto &K_child = K_children[l];
    //                     if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
    //                         // soit k_child est lr soit il l'est pas ;
    //                         if (K_child->is_low_rank()) {
    //                             Matrix<CoefficientPrecision> u  = K_child->get_low_rank_data()->Get_U();
    //                             Matrix<CoefficientPrecision> v  = K_child->get_low_rank_data()->Get_V();
    //                             Matrix<CoefficientPrecision> Hu = H->hmat_lr(u);
    //                             sr.push_back(Hu);
    //                             sr.push_back(v);
    //                             of.push_back(H->get_target_cluster().get_offset());
    //                             of.push_back(K_child->get_source_cluster().get_offset());
    //                         } else {
    //                             sh.push_back(H);
    //                             sh.push_back(K_child.get());
    //                         }
    //                     }
    //                 }
    //             }
    //         } else if (H_children.size() > 0 and K_children.size() == 0) {
    //             // K is dense sinson ce serait dans sr
    //             if (source_size0 < K->get_source_cluster().get_size()) {
    //                 std::cout << "mauvais restric , bloc dense insécable à doite" << std::endl;
    //             } else {

    //                 for (int l = 0; l < H_children.size(); ++l) {
    //                     auto &H_child = H_children[l];
    //                     if (H_child->get_target_cluster().get_size() == target_size0 and H_child->get_target_cluster().get_offset() == target_offset0) {
    //                         if (H_child->is_low_rank()) {
    //                             Matrix<CoefficientPrecision> u  = H_child->get_low_rank_data()->Get_U();
    //                             Matrix<CoefficientPrecision> v  = H_child->get_low_rank_data()->Get_V();
    //                             Matrix<CoefficientPrecision> vK = K->lr_hmat(v);
    //                             sr.push_back(u);
    //                             sr.push_back(vK);

    //                             of.push_back(H_child->get_target_cluster().get_offset());
    //                             of.push_back(K->get_source_cluster().get_offset());
    //                         } else {
    //                             sh.push_back(H_child.get());
    //                             sh.push_back(K);
    //                         }
    //                     }
    //                 }
    //             }
    //         } else if ((H_children.size() == 0) and (K_children.size() == 0)) {
    //             // H and K are dense sinon elle serait dans sr
    //             // on peut pas les couper
    //             std::cout << " mauvaise restriction : matrice insécable a gauche et a droite" << std::endl;
    //         }
    //     }
    //     SumExpression res(sr, sh, of, target_size0, target_offset0, source_size0, source_offset0);
    //     return res;
    // }
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
                                            // std::cout << vh.nb_rows() << ',' << vh.nb_cols() << '/' << uk.nb_rows() << ',' << uk.nb_cols() << '/' << vk.nb_rows() << ',' << vk.nb_cols() << std::endl;
                                            // std::cout << " ici" << std::endl;
                                            // std::cout << "norm " << normFrob(vh) << ',' << normFrob(uk) << ',' << normFrob(vk) << std::endl;
                                            Matrix<CoefficientPrecision> v = vh * uk * vk;
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
                                    for (int l = 0; l < u.nb_cols(); ++l) {
                                        u_row_major[k * u.nb_cols() + l] = u.data()[k + l * u.nb_rows()];
                                    }
                                }
                                Matrix<CoefficientPrecision> hu(H->get_target_cluster().get_size(), u.nb_cols());

                                H->add_matrix_product_row_major('N', 1.0, u_row_major.data(), 0.0, hu.data(), u.nb_cols());
                                Matrix<CoefficientPrecision> Hu(H->get_target_cluster().get_size(), u.nb_cols());
                                for (int k = 0; k < H->get_target_cluster().get_size(); ++k) {
                                    for (int l = 0; l < u.nb_cols(); ++l) {
                                        Hu.data()[l * H->get_target_cluster().get_size() + k] = hu.data()[k * u.nb_cols() + l];
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
                                    for (int l = 0; l < v.nb_cols(); +l) {
                                        v_row_major[k * v.nb_cols() + l] = v_transp.data()[k + l * v.nb_rows()];
                                    }
                                }
                                Matrix<CoefficientPrecision> vk(K->get_source_cluster().get_size(), v.nb_rows());
                                K->add_matrix_product_row_major('T', 1.0, v_row_major.data(), 0.0, vk.data(), v.nb_rows());
                                Matrix<CoefficientPrecision> vK(v.nb_rows(), K->get_source_cluster().get_size());
                                for (int k = 0; k < v.nb_rows(); ++k) {
                                    for (int l = 0; l < K->get_source_cluster().get_size(); ++l) {
                                        vK.data()[k * K->get_source_cluster().get_size() + l] = vk.data()[k + l * v.nb_rows()];
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

    ///////////////////////////////////////////////
    //// Restrict sans is_restricticle
    ////////////////////////////////

    const SumExpression
    Restrict_new(int target_size0, int target_offset0, int source_size0, int source_offset0) const {
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
                                            std::cout << "!!!!!!!!!!!!!!!!!!!" << std::endl;
                                            Matrix<CoefficientPrecision> u  = H_child->get_low_rank_data()->Get_U();
                                            Matrix<CoefficientPrecision> v  = H_child->get_low_rank_data()->Get_V();
                                            Matrix<CoefficientPrecision> vk = K_child->lr_hmat(v);
                                            sr.push_back(u);
                                            sr.push_back(vk);
                                            of.push_back(H_child->get_target_cluster().get_offset());
                                            of.push_back(K_child->get_source_cluster().get_offset());
                                        } else if (!(H_child->is_low_rank()) and (K_child->is_low_rank())) {
                                            std::cout << "!!!!!" << std::endl;
                                            Matrix<CoefficientPrecision> u  = K_child->get_low_rank_data()->Get_U();
                                            Matrix<CoefficientPrecision> v  = K_child->get_low_rank_data()->Get_V();
                                            Matrix<CoefficientPrecision> hu = H_child->hmat_lr(u);
                                            sr.push_back(hu);
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
                    Matrix<CoefficientPrecision> Hdense = *H->get_dense_data();
                    Matrix<CoefficientPrecision> Hrestr(target_size0, H->get_source_cluster().get_size());
                    for (int i = 0; i < target_size0; ++i) {
                        for (int j = 0; j < H->get_source_cluster().get_size(); ++j) {
                            Hrestr(i, j) = Hdense(i + target_offset0 - H->get_target_cluster().get_offset(), j);
                        }
                    }
                    for (int l = 0; l < K_children.size(); ++l) {
                        auto &K_child = K_children[l];
                        if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
                            // soit k_child est lr soit il l'est pas ;
                            if (K_child->is_low_rank()) {
                                Matrix<CoefficientPrecision> u  = K_child->get_low_rank_data()->Get_U();
                                Matrix<CoefficientPrecision> v  = K_child->get_low_rank_data()->Get_V();
                                Matrix<CoefficientPrecision> Hu = H->hmat_lr(u);
                                sr.push_back(Hu);
                                sr.push_back(v);
                                of.push_back(H->get_target_cluster().get_offset());
                                of.push_back(K_child->get_source_cluster().get_offset());
                            } else {
                                sr.push_back(Hrestr);
                                Matrix<CoefficientPrecision> kdense(H->get_source_cluster().get_size(), source_size0);
                                copy_to_dense(*K_child, kdense.data());
                                sr.push_back(kdense);
                                // sr.push_back(*K_child->get_dense_data());
                                of.push_back(target_offset0);
                                of.push_back(source_offset0);
                            }
                        }
                    }

                } else {
                    // donc la normalement K a juste 2 fils et on prend celui qui a le bon sigma
                    for (int l = 0; l < K_children.size(); ++l) {
                        auto &K_child = K_children[l];
                        if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
                            // soit k_child est lr soit il l'est pas ;
                            if (K_child->is_low_rank()) {
                                Matrix<CoefficientPrecision> u  = K_child->get_low_rank_data()->Get_U();
                                Matrix<CoefficientPrecision> v  = K_child->get_low_rank_data()->Get_V();
                                Matrix<CoefficientPrecision> Hu = H->hmat_lr(u);
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
                    Matrix<CoefficientPrecision> Kdense = *K->get_dense_data();
                    Matrix<CoefficientPrecision> Krestr(K->get_target_cluster().get_size(), source_size0);
                    for (int i = 0; i < K->get_target_cluster().get_size(); ++i) {
                        for (int j = 0; j < source_size0; ++j) {
                            Krestr(i, j) = Kdense(i, j + source_offset0 - K->get_source_cluster().get_offset());
                        }
                    }
                    for (int l = 0; l < H_children.size(); ++l) {
                        auto &H_child = H_children[l];
                        if (H_child->get_target_cluster().get_size() == target_size0 and H_child->get_target_cluster().get_offset() == target_offset0) {
                            if (H_child->is_low_rank()) {
                                Matrix<CoefficientPrecision> u  = H_child->get_low_rank_data()->Get_U();
                                Matrix<CoefficientPrecision> v  = H_child->get_low_rank_data()->Get_V();
                                Matrix<CoefficientPrecision> vK = K->lr_hmat(v);
                                sr.push_back(u);
                                sr.push_back(vK);

                                of.push_back(H_child->get_target_cluster().get_offset());
                                of.push_back(K->get_source_cluster().get_offset());
                            } else {
                                Matrix<CoefficientPrecision> hdense(target_size0, K->get_target_cluster().get_size());
                                copy_to_dense(*H_child, hdense.data());
                                sr.push_back(hdense);
                                sr.push_back(Krestr);
                                of.push_back(target_offset0);
                                of.push_back(source_offset0);
                            }
                        }
                    }
                } else {

                    for (int l = 0; l < H_children.size(); ++l) {
                        auto &H_child = H_children[l];
                        if (H_child->get_target_cluster().get_size() == target_size0 and H_child->get_target_cluster().get_offset() == target_offset0) {
                            if (H_child->is_low_rank()) {
                                Matrix<CoefficientPrecision> u  = H_child->get_low_rank_data()->Get_U();
                                Matrix<CoefficientPrecision> v  = H_child->get_low_rank_data()->Get_V();
                                Matrix<CoefficientPrecision> vK = K->lr_hmat(v);
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
                Matrix<CoefficientPrecision> Hrestr(target_size0, H->get_source_cluster().get_size());
                auto Hdense = *H->get_dense_data();
                auto Kdense = *K->get_dense_data();
                Matrix<CoefficientPrecision> Krestr(K->get_target_cluster().get_size(), source_size0);
                for (int i = 0; i < target_size0; ++i) {
                    for (int j = 0; j < H->get_source_cluster().get_size(); ++j) {
                        Hrestr(i, j) = Hdense(i + target_offset0 - H->get_target_cluster().get_offset(), j);
                    }
                }
                for (int i = 0; i < K->get_target_cluster().get_size(); ++i) {
                    for (int j = 0; j < source_size0; ++j) {
                        Krestr(i, j) = Kdense(i, j + source_offset0 - K->get_source_cluster().get_offset());
                    }
                }
                sr.push_back(Hrestr);
                sr.push_back(Krestr);
                of.push_back(target_offset0);
                of.push_back(source_offset0);
            }
        }
        SumExpression res(sr, sh, of, target_size0, target_offset0, source_size0, source_offset0);
        return res;
    }
    // const SumExpression new_Restrict(int target_size0, int target_offset0, int source_size0, int source_offset0) const {
    //     std::vector<const HMatrixType *> sh;
    //     std::vector<Matrix<CoefficientPrecision>> sr;
    //     auto of = offset;
    //     for (auto s : Sr) {
    //         sr.push_back(s);
    //     }
    //     for (int rep = 0; rep < Sh.size() / 2; ++rep) {
    //         auto &H          = Sh[2 * rep];
    //         auto &K          = Sh[2 * rep + 1];
    //         auto &H_children = H->get_children();
    //         auto &K_children = K->get_children();
    //         if ((H_children.size() > 0) and (K_children.size() > 0)) {
    //             for (auto &H_child : H_children) {
    //                 if ((H_child->get_target_cluster().get_size() == target_size0) and (H_child->get_target_cluster().get_offset() == target_offset0)) {
    //                     for (auto &K_child : K_children) {
    //                         if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
    //                             if ((K_child->get_target_cluster().get_size() == H_child->get_source_cluster().get_size())
    //                                 and (K_child->get_target_cluster().get_offset() == H_child->get_source_cluster().get_offset())) {
    //                                 if (H_child->is_low_rank() or K_child->is_low_rank()) {
    //                                     if (H_child->is_low_rank() and K_child->is_low_rank()) {
    //                                         Matrix<CoefficientPrecision> uh = H_child->get_low_rank_data()->Get_U();
    //                                         Matrix<CoefficientPrecision> vh = H_child->get_low_rank_data()->Get_V();
    //                                         Matrix<CoefficientPrecision> uk = K_child->get_low_rank_data()->Get_U();
    //                                         Matrix<CoefficientPrecision> vk = K_child->get_low_rank_data()->Get_V();
    //                                         Matrix<CoefficientPrecision> v  = vh * uk * vk;
    //                                         sr.push_back(uh);
    //                                         sr.push_back(v);
    //                                         of.push_back(H_child->get_target_cluster().get_offset());
    //                                         of.push_back(K_child->get_source_cluster().get_offset());
    //                                     } else if ((H_child->is_low_rank()) and !(K_child->is_low_rank())) {
    //                                         // std::cout << "!2" << std::endl;
    //                                         Matrix<CoefficientPrecision> u = H_child->get_low_rank_data()->Get_U();
    //                                         Matrix<CoefficientPrecision> v = H_child->get_low_rank_data()->Get_V();
    //                                         // Matrix<CoefficientPrecision> vk = K_child->mult(v, 'T');
    //                                         Matrix<CoefficientPrecision> temp(K_child->get_target_cluster().get_size(), K_child->get_source_cluster().get_size());
    //                                         copy_to_dense(*K_child, temp.data());
    //                                         Matrix<CoefficientPrecision> vk = v * temp;
    //                                         // std::cout << "size avant " << sr.size() << std::endl;
    //                                         if (sr.size() > 0) {
    //                                             // std::cout << "2 avant:" << sr[0].nb_rows() << ',' << sr[0].nb_cols() << ',' << sr[1].nb_rows() << ',' << sr[1].nb_cols() << std::endl;
    //                                         }
    //                                         // std::cout << "on push " << u.nb_rows() << ',' << u.nb_cols() << ',' << vk.nb_rows() << ',' << vk.nb_cols() << std::endl;
    //                                         sr.push_back(u);
    //                                         sr.push_back(vk);
    //                                         // std::cout << "2 apres :" << sr[0].nb_rows() << ',' << sr[0].nb_cols() << ',' << sr[1].nb_rows() << ',' << sr[1].nb_cols() << std::endl;
    //                                         // std::cout << "size apres " << sr.size() << std::endl;
    //                                         of.push_back(H_child->get_target_cluster().get_offset());
    //                                         of.push_back(K_child->get_source_cluster().get_offset());
    //                                     } else if (!(H_child->is_low_rank()) and (K_child->is_low_rank())) {
    //                                         // std::cout << "!3" << std::endl;
    //                                         Matrix<CoefficientPrecision> u = K_child->get_low_rank_data()->Get_U();
    //                                         Matrix<CoefficientPrecision> v = K_child->get_low_rank_data()->Get_V();
    //                                         // Matrix<CoefficientPrecision> hu = H_child->mult(u, 'N');
    //                                         Matrix<CoefficientPrecision> temp(H_child->get_target_cluster().get_size(), H_child->get_source_cluster().get_size());
    //                                         copy_to_dense(*H_child, temp.data());
    //                                         Matrix<CoefficientPrecision> hu = temp * u;
    //                                         // std::cout << hu.nb_rows() << ',' << hu.nb_cols() << ',' << v.nb_rows() << ',' << v.nb_cols() << std::endl;
    //                                         sr.push_back(hu);
    //                                         sr.push_back(v);
    //                                         // std::cout << "3:" << sr[0].nb_rows() << ',' << sr[0].nb_cols() << ',' << sr[1].nb_rows() << ',' << sr[1].nb_cols() << std::endl;
    //                                         of.push_back(H_child->get_target_cluster().get_offset());
    //                                         of.push_back(K_child->get_source_cluster().get_offset());
    //                                     }
    //                                 } else {
    //                                     sh.push_back(H_child.get());
    //                                     sh.push_back(K_child.get());
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         } else if (H_children.size() == 0 and K_children.size() > 0) {
    //             // H is dense otherwise HK would be in SR
    //             if (target_size0 < H->get_target_cluster().get_size()) {
    //                 std::cout << " mauvais restrict , bloc dense insécablble a gauche" << std::endl;
    //             } else {
    //                 Matrix<CoefficientPrecision> H_dense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
    //                 copy_to_dense(*H, H_dense.data());
    //                 // donc la normalement K a juste 2 fils et on prend celui qui a le bon sigma
    //                 for (int l = 0; l < K_children.size(); ++l) {
    //                     auto &K_child = K_children[l];
    //                     if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
    //                         // soit k_child est lr soit il l'est pas ;
    //                         if (K_child->is_low_rank()) {
    //                             // std::cout << "!4" << std::endl;
    //                             Matrix<CoefficientPrecision> u  = K_child->get_low_rank_data()->Get_U();
    //                             Matrix<CoefficientPrecision> v  = K_child->get_low_rank_data()->Get_V();
    //                             Matrix<CoefficientPrecision> Hu = H_dense * u;
    //                             sr.push_back(Hu);
    //                             sr.push_back(v);
    //                             // std::cout << "4 :" << sr[0].nb_rows() << ',' << sr[0].nb_cols() << ',' << sr[1].nb_rows() << ',' << sr[1].nb_cols() << std::endl;
    //                             of.push_back(H->get_target_cluster().get_offset());
    //                             of.push_back(K_child->get_source_cluster().get_offset());
    //                         } else {
    //                             sh.push_back(H);
    //                             sh.push_back(K_child.get());
    //                         }
    //                     }
    //                 }
    //             }
    //         } else if (H_children.size() > 0 and K_children.size() == 0) {
    //             // K is dense sinson ce serait dans sr
    //             if (source_size0 < K->get_source_cluster().get_size()) {
    //                 std::cout << "mauvais restric , bloc dense insécable à doite" << std::endl;
    //             } else {
    //                 Matrix<CoefficientPrecision> K_dense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
    //                 copy_to_dense(*K, K_dense.data());
    //                 for (int l = 0; l < H_children.size(); ++l) {
    //                     auto &H_child = H_children[l];
    //                     if (H_child->get_target_cluster().get_size() == target_size0 and H_child->get_target_cluster().get_offset() == target_offset0) {
    //                         if (H_child->is_low_rank()) {
    //                             // std::cout << "!5" << std::endl;
    //                             Matrix<CoefficientPrecision> u  = H_child->get_low_rank_data()->Get_U();
    //                             Matrix<CoefficientPrecision> v  = H_child->get_low_rank_data()->Get_V();
    //                             Matrix<CoefficientPrecision> vK = v * K_dense;
    //                             sr.push_back(u);
    //                             sr.push_back(vK);
    //                             // std::cout << "5 :" << sr[0].nb_rows() << ',' << sr[0].nb_cols() << ',' << sr[1].nb_rows() << ',' << sr[1].nb_cols() << std::endl;
    //                             of.push_back(H_child->get_target_cluster().get_offset());
    //                             of.push_back(K->get_source_cluster().get_offset());
    //                         } else {
    //                             sh.push_back(H_child.get());
    //                             sh.push_back(K);
    //                         }
    //                     }
    //                 }
    //             }
    //         } else if ((H_children.size() == 0) and (K_children.size() == 0)) {
    //             // H and K are dense sinon elle serait dans sr
    //             // on peut pas les couper
    //             std::cout << " mauvaise restriction : matrice insécable a gauche et a droite" << std::endl;
    //         }
    //     }
    //     // std::cout << "a la fin" << std::endl;
    //     // if (sr.size() > 0) {
    //     //     std::cout << sr[0].nb_rows() << ',' << sr[0].nb_cols() << ',' << sr[1].nb_rows() << ',' << sr[1].nb_cols() << std::endl;
    //     // }
    //     SumExpression res(sr, sh, of, target_size0, target_offset0, source_size0, source_offset0);
    //     // if (sr.size() > 0) {
    //     //     std::cout << "test" << res.get_sr()[0].nb_rows() << ',' << res.get_sr()[0].nb_cols() << ',' << res.get_sr()[1].nb_rows() << ',' << res.get_sr()[1].nb_cols() << std::endl;
    //     // }
    //     return res;
    // }
};
} // namespace htool

#endif
