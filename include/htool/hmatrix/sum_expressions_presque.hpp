#ifndef SUM_EXPRESSIONS_HPP
#define SUM_EXPRESSIONS_HPP
#include "../basic_types/vector.hpp"

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
    // struct LR {
    //     LRtype lrmat;
    //     int offset_lr_target;
    //     int offset_lr_source;
    //     LR(const LRtype &mat, int target, int source) : lrmat(mat), offset_lr_target(target), offset_lr_source(source) {}
    // };
    struct LR {
        LRtype &lrmat;
        int offset_lr_target;
        int offset_lr_source;
        LR(LRtype &mat, int target, int source) : lrmat(mat), offset_lr_target(target), offset_lr_source(source) {}
    };
    std::vector<const LR *> Sr;
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

    SumExpression(std::vector<const LR *> sr, std::vector<const HMatrixType *> sh, int target_size0, int target_offset0, int source_size0, int source_offset0) {
        Sr            = sr;
        Sh            = sh;
        target_size   = target_size0;
        target_offset = target_offset0;
        source_size   = source_size0;
        source_offset = source_offset0;
    }

    std::vector<const LR *> get_sr() { return Sr; }
    std::vector<const HMatrixType *> get_sh() { return Sh; }
    int get_nr() { return target_size; }
    int get_nc() { return source_size; }
    int get_target_offset() { return target_offset; }
    int get_source_offset() { return source_offset; }

    std::vector<CoefficientPrecision> prod(const std::vector<CoefficientPrecision> &x) {
        // std::cout << target_size << ',' << source_size << std::endl;
        // std::cout << x.size() << std::endl;
        std::vector<CoefficientPrecision> y(target_size, 0.0);
        for (auto &low_rank_leaf_hmatrix : Sr) {
            std::cout << "on  lr * x" << std::endl;
            auto &lr = low_rank_leaf_hmatrix->lrmat;
            std::cout << "on a pris lr " << lr.nb_rows() << ',' << lr.nb_cols() << ',' << lr.rank_of() << std::endl;
            auto uu = lr.Get_U();
            std::cout << target_size << ',' << source_size << ',' << uu.nb_cols() << std::endl;
            Matrix<CoefficientPrecision> u(target_size, lr.rank_of());
            Matrix<CoefficientPrecision> v(lr.rank_of(), source_size);
            CoefficientPrecision *ptru = new CoefficientPrecision[target_size * lr.rank_of()];
            CoefficientPrecision *ptrv = new CoefficientPrecision[lr.rank_of() * source_size];
            // std::cout << "création pointeur" << std::endl;
            // std::cout << "u" << target_size << ',' << target_offset << ',' << lr.rank_of() << std::endl;
            for (int i = 0; i < target_size; ++i) {
                for (int j = 0; j < lr.rank_of(); ++j) {
                    ptru[i + target_size * j] = lr.get_U(i + target_offset - low_rank_leaf_hmatrix->offset_lr_target, j);
                }
            }
            // std::cout << " u assigned " << std::endl;
            for (int i = 0; i < lr.rank_of(); ++i) {
                for (int j = 0; j < source_size; ++j) {
                    ptrv[i + lr.rank_of() * j] = lr.get_V(i, j + source_offset - low_rank_leaf_hmatrix->offset_lr_source);
                }
            }
            // std::cout << "v assigned" << std::endl;
            u.assign(target_size, lr.rank_of(), ptru, true);
            v.assign(lr.rank_of(), source_size, ptrv, true);
            // std::cout << "matrix assigned" << std::endl;
            y = y + u * (v * x);
            delete[] ptru;
            delete[] ptrv;
            // std::cout << "y+=" << std::endl;
        }
        // std::cout << " ok pour les lr " << std::endl;
        for (int k = 0; k < Sh.size() / 2; ++k) {
            const HMatrixType &H = *Sh[2 * k];
            const HMatrixType &K = *Sh[2 * k + 1];
            std::vector<CoefficientPrecision> y_temp(K.get_target_cluster().get_size(), 0.0);
            K.add_vector_product('N', 1.0, x.data(), 0.0, y_temp.data());
            H.add_vector_product('N', 1.0, y_temp.data(), 1.0, y.data());
        }
        return (y);
    }

    CoefficientPrecision get_coeff(const int &i, const int &j) {
        std::vector<CoefficientPrecision> xj(source_size, 0.0);
        xj[j]                               = 1.0;
        std::vector<CoefficientPrecision> y = this->prod(xj);
        return y[i];
    }

    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        auto H = *this;
        // std::cout << M << ',' << N << ',' << row_offset << ',' << col_offset << std::endl;
        // std::cout << H.target_size << ',' << H.source_size << std::endl;
        for (int k = 0; k < M; ++k) {
            for (int l = 0; l < N; ++l) {
                std::vector<CoefficientPrecision> xl(source_size, 0.0);
                xl[l]                               = 1.0;
                std::vector<CoefficientPrecision> y = H.prod(xl);
                // std::cout << "copy sub matrix  , prod " << k << ',' << l << std::endl;
                ptr[k + M * l] = y[k];
            }
        }
    }

    // void copy_submatrix(int M, int N, int offset_row, int offset_col, CoefficientPrecision *ptr) const override {
    //     CoefficientPrecision *ptr_temp = new CoefficientPrecision[M * N];
    //     for (int k = 0; k < Sr.size(); ++k) {
    //         auto my_lr                 = Sr[0];
    //         auto lr                    = my_lr->lrmat;
    //         auto oft                   = my_lr->offset_lr_target;
    //         auto ofs                   = my_lr->offset_lr_source;
    //         CoefficientPrecision *ptru = new CoefficientPrecision[M * lr.rank_of()];
    //         CoefficientPrecision *ptrv = new CoefficientPrecision[lr.rank_of() * N];
    //         for (int i = 0; i < M; ++i) {
    //             for (int j = 0; j < lr.rank_of(); ++j) {
    //                 ptru[i + M * j] = lr.get_U(i + offset_row - oft, j);
    //             }
    //         }
    //         for (int i = 0; i < lr.rank_of(); ++i) {
    //             for (int j = 0; j < N; ++j) {
    //                 ptrv[i + lr.rank_of() * j] = lr.get_V(i, j + offset_col - ofs);
    //             }
    //         }
    //         Matrix<CoefficientPrecision> u(M, lr.rank_of());
    //         Matrix<CoefficientPrecision> v(lr.rank_of(), N);
    //         u.assign(M, lr.rank_of(), ptru, true);
    //         v.assign(lr.rank_of(), N, ptrv, true);
    //         auto uv  = u * v;
    //         auto dat = uv.data();
    //         for (int l = 0; l < M * N; ++l) {
    //             ptr_temp[l] = ptr_temp[l] + dat[l];
    //         }
    //         delete[] ptru;
    //         delete[] ptrv;
    //     }
    //     for (int k = 0; k < Sh.size() / 2; ++k) {
    //         auto H = Sh[2 * k];
    //         auto K = Sh[2 * k + 1];
    //         // pour avoir le coeff i j on fait avec des mult matrice vecteur
    //         // M * (0.....1.........0) = M(: , j) -> (à..1.....0)^T *M*(0......1...0)= M(i,j) avec M= H*K
    //         for (int j = 0; j < N; ++j) {
    //             std::vector<CoefficientPrecision> ej(K->get_source_cluster().get_size(), 0.0);
    //             ej[j + offset_col - K->get_source_cluster().get_offset()] = 1.0;
    //             std::vector<CoefficientPrecision> y_temp(K->get_target_cluster().get_size(), 0.0);
    //             std::vector<CoefficientPrecision> y(H->get_target_cluster().get_size(), 0.0);
    //             K->add_vector_product('N', 1.0, ej.data(), 0.0, y_temp.data());
    //             H->add_vector_product('N', 1.0, y_temp.data(), 0.0, y.data());
    //             for (int i = 0; i < M; ++i) {
    //                 ptr_temp[i + M * j] = ptr_temp[i + M * j] + y[i + offset_row - H->get_target_cluster().get_offset()];
    //             }
    //         }
    //     }
    //     std::copy_n(ptr_temp, M * N, ptr);
    //     delete[] ptr_temp;
    // }

    // void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
    //     Matrix<CoefficientPrecision> temp(M, N);
    //     for (int i = 0; i < Sr.size(); ++i) {
    //         auto L  = Sr[i];
    //         auto lr = L->lrmat;
    //         int oft = L->offset_lr_target;
    //         int ofs = L->offset_lr_source;
    //         auto U  = lr.Get_U();
    //         auto V  = lr.Get_V();
    //         Matrix<CoefficientPrecision> tempu(M, U.nb_cols());
    //         Matrix<CoefficientPrecision> tempv(V.nb_rows(), N);
    //         CoefficientPrecision *ptru = new CoefficientPrecision[M * U.nb_cols()];
    //         CoefficientPrecision *ptrv = new CoefficientPrecision[V.nb_rows() * N];
    //         for (int j = 0; j < M; j++) {
    //             for (int k = 0; k < U.nb_cols(); k++) {
    //                 // cout<< "ici" << k<<','<< j << ',' <<rows[j]<< ',' << cols[k]<< ','<< mat.nb_rows()<< ',' << mat.nb_cols() << endl;
    //                 ptru[j + M * k] = U(j + row_offset - oft, k);
    //             }
    //         }
    //         for (int j = 0; j < V.nb_rows(); j++) {
    //             for (int k = 0; k < N; k++) {
    //                 // cout<< "ici" << k<<','<< j << ',' <<rows[j]<< ',' << cols[k]<< ','<< mat.nb_rows()<< ',' << mat.nb_cols() << endl;
    //                 ptrv[j + V.nb_cols() * k] = V(j, k + col_offset - ofs);
    //             }
    //         }
    //         tempu.assign(M, U.nb_cols(), ptru, true);
    //         tempv.assign(V.nb_rows(), N, ptrv, true);
    //         temp = temp + tempu * tempv;
    //     }
    //     for (int i = 0; i < Sh.size() / 2; ++i) {
    //         auto H                      = Sh[2 * i];
    //         auto K                      = Sh[2 * i + 1];
    //         CoefficientPrecision *ptrhk = new CoefficientPrecision[M * N];
    //         for (int k = 0; k < M; ++k) {
    //             for (int l = 0; l < N; ++l) {
    //                 std::vector<double> y(N, 0.0);
    //                 std::vector<double> yj(N, 0.0);
    //                 yj[l] = 1;
    //                 std::vector<CoefficientPrecision> y_temp(K->get_target_cluster().get_size(), 0.0);
    //                 K->add_vector_product('N', 1.0, yj.data(), 0.0, y_temp.data());
    //                 H->add_vector_product('N', 1.0, y_temp.data(), 1.0, y.data());
    //                 ptrhk[k + M * l] = y[k];
    //             }
    //         }
    //         Matrix<CoefficientPrecision> HK(M, N);
    //         HK.assign(M, N, ptrhk, true);
    //         temp = temp + HK;
    //     }
    //     std::copy_n(ptr, M * N, temp.data());
    // }

    std::vector<int>
    is_restrictible() {
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

    // ///////////////////////////////////////////////////////
    // // RESTRICT
    // /////////////////////////////////////////////

    // On est censé appeler restrict (tau , sigma) seulementsi il y a pas de dense plus grande que tau a gauche ou de dense plus grande que sigma a droite
    SumExpression Restrict(int target_size0, int target_offset0, int source_size0, int source_offset0) const {
        std::vector<const HMatrixType *> sh;
        std::vector<const LR *> sr;
        sr = Sr;
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
                                            auto uh = H_child->get_low_rank_data()->Get_U();
                                            auto vh = H_child->get_low_rank_data()->Get_V();
                                            auto uk = K_child->get_low_rank_data()->Get_U();
                                            auto vk = K_child->get_low_rank_data()->Get_V();
                                            auto v  = vh * uk * vk;
                                            LRtype lr(uh, v);
                                            LR lrprod(lr, target_offset0, source_offset0);
                                            // lrprod.lrmat(lr);
                                            // lrprod.offset_lr_target = target_offset0;
                                            // lrprod.offset_lr_source = source_offset0;
                                            sr.push_back(&lrprod);
                                            std::cout << "on vient de push un lr " << lr.nb_rows() << ',' << lr.nb_cols() << ',' << lr.rank_of() << std::endl;

                                        } else if ((H_child->is_low_rank()) and !(K_child->is_low_rank())) {
                                            auto u  = H_child->get_low_rank_data()->Get_U();
                                            auto v  = H_child->get_low_rank_data()->Get_V();
                                            auto vk = K_child->mult(v, 'T');
                                            LRtype lr(u, vk);
                                            LR lrprod(lr, target_offset0, source_offset0);
                                            // lrprod.lrmat            = lr;
                                            // lrprod.offset_lr_target = target_offset0;
                                            // lrprod.offset_lr_offset = source_offset0;
                                            sr.push_back(&lrprod);
                                            std::cout << "on vient de push un lr " << lr.nb_rows() << ',' << lr.nb_cols() << ',' << lr.rank_of() << std::endl;

                                        } else if (!(H_child->is_low_rank()) and (K_child->is_low_rank())) {
                                            auto u  = K_child->get_low_rank_data()->Get_U();
                                            auto v  = K_child->get_low_rank_data()->Get_V();
                                            auto hu = H_child->mult(u, 'N');
                                            LRtype lr(hu, v);
                                            LR lrprod(lr, target_offset0, source_offset0);
                                            // lrprod.lrmat             = lr;
                                            // lrprod.aoffset_lr_target = target_offset0;
                                            // lrprod.offset_lr_source  = source_offset0;
                                            sr.push_back(&lrprod);
                                            std::cout << "on vient de push un lr " << lr.nb_rows() << ',' << lr.nb_cols() << ',' << lr.rank_of() << std::endl;
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
                    Matrix<CoefficientPrecision> H_dense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                    copy_to_dense(*H, H_dense.data());
                    // donc la normalement K a juste 2 fils et on prend celui qui a le bon sigma
                    for (int l = 0; l < K_children.size(); ++l) {
                        auto &K_child = K_children[l];
                        if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
                            // soit k_child est lr soit il l'est pas ;
                            if (K_child->is_low_rank()) {
                                auto u  = K_child->get_low_rank_data()->Get_U();
                                auto v  = K_child->get_low_rank_data()->Get_V();
                                auto Hu = H_dense * u;
                                std::cout << "ceci est un test" << std::endl;
                                std::cout << H_dense.nb_rows() << ',' << H_dense.nb_cols() << ',' << u.nb_rows() << ',' << u.nb_cols() << std::endl;
                                std::cout << normFrob(Hu * v) << std::endl;
                                LRtype lr(Hu, v);
                                std::cout << "restrict " << std::endl;
                                std::cout << lr.Get_U().nb_rows() << ',' << lr.Get_V().nb_cols() << std::endl;
                                LR lrprod(lr, target_offset0, source_offset0);
                                sr.push_back(&lrprod);
                                std::cout << "on vient de push un lr " << lr.nb_rows() << ',' << lr.nb_cols() << ',' << lr.rank_of() << std::endl;

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
                    Matrix<CoefficientPrecision> K_dense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                    copy_to_dense(*K, K_dense.data());
                    for (int l = 0; l < H_children.size(); ++l) {
                        auto &H_child = H_children[l];
                        if (H_child->get_target_cluster().get_size() == target_size0 and H_child->get_target_cluster().get_offset() == target_offset0) {
                            if (H_child->is_low_rank()) {
                                auto u  = H_child->get_low_rank_data()->Get_U();
                                auto v  = H_child->get_low_rank_data()->Get_V();
                                auto vK = v * K_dense;
                                LRtype lr(u, vK);
                                LR lrprod(lr, target_offset0, source_offset0);
                                sr.push_back(&lrprod);
                                std::cout << "on vient de push un lr " << lr.nb_rows() << ',' << lr.nb_cols() << ',' << lr.rank_of() << std::endl;

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
        SumExpression res(sr, sh, target_size0, target_offset0, source_size0, source_offset0);
        return res;
    }

    // /////////////////////////////////////////////////////
    // //// EVALUATE
    // //////////////////////////////////////////////

    Matrix<CoefficientPrecision>
    Evaluate() {
        Matrix<CoefficientPrecision> res(target_size, source_size);
        for (int k = 0; k < Sr.size() / 2; ++k) {
            auto H = Sr[2 * k];
            auto K = Sr[2 * k + 1];
            if (H->is_low_rank() and K->is_low_rank()) {
                std::cout << 1 << std::endl;
                Matrix<CoefficientPrecision> U_H, V_H, U_K, V_K;
                U_H = H->get_low_rank_data()->Get_U();
                V_H = H->get_low_rank_data()->Get_V();
                U_K = K->get_low_rank_data()->Get_U();
                V_K = K->get_low_rank_data()->Get_V();
                Matrix<CoefficientPrecision> U_restr(target_size, U_H.nb_cols());
                Matrix<CoefficientPrecision> V_restr(V_K.nb_rows(), source_size);
                CoefficientPrecision *ptr_U = new CoefficientPrecision(target_size * U_H.nb_cols());
                CoefficientPrecision *ptr_V = new CoefficientPrecision(source_size * V_K.nb_rows());
                for (int j = 0; j < U_H.nb_cols(); ++j) {
                    for (int i = 0; i < target_size; ++i) {
                        ptr_U[i + j * target_size] = U_H(i + target_offset - H->get_target_cluster().get_offset(), j);
                    }
                }
                for (int j = 0; j < source_size; ++j) {
                    for (int i = 0; i < K->get_target_cluster().get_size(); ++i) {
                        ptr_V[i + j * K->get_target_cluster().get_size()] = V_K(i, j + source_offset - K->get_source_cluster().get_offset());
                    }
                }
                U_restr.assign(target_size, U_H.nb_cols(), ptr_U, true);
                V_restr.assign(V_K.nb_rows(), source_size, ptr_V, true);
                Matrix<CoefficientPrecision> temp1(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                Matrix<CoefficientPrecision> temp2(U_H.nb_cols(), K->get_source_cluster().get_size());
                // U_K.add_matrix_product('N', 1.0, V_K.data(), 0.0, temp1.data(), 1);
                // V_H.add_matrix_product('N', 1.0, temp1.data(), 0.0, temp2.data(), 1);
                // U_H.add_matrix_product('N', 1.0, temp2.data(), 1.0, res.data(), 1);
                res = res + U_H * V_H * U_K * V_K;
                delete[] ptr_U;
                delete[] ptr_V;
            } else if ((H->is_low_rank()) and !(K->is_low_rank())) {
                std::cout << 2 << std::endl;

                auto U = K->get_low_rank_data()->Get_U();
                auto V = K->get_low_rank_data()->Get_V();
                Matrix<CoefficientPrecision> U_restr(target_size, U.nb_cols());
                CoefficientPrecision *ptr_U = new CoefficientPrecision[target_size * U.nb_cols()];
                for (int j = 0; j < U.nb_cols(); ++j) {
                    for (int i = 0; i < target_size; ++i) {
                        ptr_U[i + j * target_size] = U(i + target_offset - H->get_target_cluster().get_offset(), j);
                    }
                }
                U_restr.assign(target_size, U.nb_cols(), ptr_U, true);
                // ligne *hmat
                Matrix<CoefficientPrecision> VK(V.nb_rows(), source_size);
                CoefficientPrecision *ptr_VK = new CoefficientPrecision[V.nb_rows() * source_size];
                for (int i = 0; i < V.nb_rows(); ++i) {
                    std::vector<CoefficientPrecision> row_Vi(V.nb_cols(), 0.0);
                    for (int j = 0; j < K->get_target_cluster().get_size(); ++j) {
                        row_Vi[j] = V(i, j + K->get_target_cluster().get_offset() - H->get_source_cluster().get_offset());
                    }
                    std::vector<CoefficientPrecision> row_i(source_size, 0.0);
                    K->add_vector_product('T', 1.0, row_Vi.data(), 0.0, row_i.data());
                    for (int j = 0; j < source_size; ++j) {
                        ptr_VK[i + j * V.nb_rows()] = row_i[j];
                    }
                }
                VK.assign(V.nb_rows(), source_size, ptr_VK, true);
                // U_restr.add_matrix_product('N', 1.0, VK.data(), 1.0, res.data(), 1);
                res = res + U_restr * VK;
                delete[] ptr_U;
                delete[] ptr_VK;
            } else if (!(H->is_low_rank()) and (K->is_low_rank())) {
                std::cout << 3 << std::endl;

                auto U = K->get_low_rank_data()->Get_U();
                auto V = K->get_low_rank_data()->Get_V();
                Matrix<CoefficientPrecision> V_restr(V.nb_rows(), source_size);
                CoefficientPrecision *ptr_V = new CoefficientPrecision[V.nb_rows() * source_size];
                for (int j = 0; j < source_size; ++j) {
                    for (int i = 0; i < V.nb_rows(); ++i) {
                        ptr_V[i + j * V.nb_rows()] = V(i, j + source_offset - K->get_source_cluster().get_offset());
                    }
                }
                V_restr.assign(V.nb_rows(), source_size, ptr_V, true);
                Matrix<CoefficientPrecision> HU(target_size, U.nb_cols());
                CoefficientPrecision *ptr_HU = new CoefficientPrecision[target_size * U.nb_cols()];
                for (int j = 0; j < U.nb_cols(); ++j) {
                    std::vector<CoefficientPrecision> col_Uj(U.nb_rows(), 0.0);
                    for (int i = 0; i < H->get_source_cluster().get_size(); ++i) {
                        col_Uj[i] = U(i + H->get_source_cluster().get_offset() - K->get_target_cluster().get_offset(), j);
                    }
                    std::vector<CoefficientPrecision> col_j(target_size, 0.0);
                    H->add_vector_product('N', 1.0, col_Uj.data(), 0.0, col_j.data());
                    for (int i = 0; i < target_size; ++i) {
                        ptr_HU[i + j * target_size] = col_j[i];
                    }
                }
                HU.assign(target_size, U.nb_cols(), ptr_HU, true);
                // HU.add_matrix_product('N', 1.0, V_restr.data(), 1.0, res.data(), 1);
                res = res + HU * V_restr;
                delete[] ptr_V;
                delete[] ptr_HU;
            }
        }
        for (int rep = 0; rep < Sh.size() / 2; ++rep) {
            std::cout << 4 << std::endl;

            auto &H = Sh[2 * rep];
            auto &K = Sh[2 * rep + 1];
            std::cout << target_size << ',' << target_offset << ',' << source_size << ',' << source_offset << std::endl;
            // Matrix<CoefficientPrecision> H_dense(10, 10);
            // std::cout << H->get_source_cluster().get_size() << std::endl;
            int size = H->get_source_cluster().get_size();
            std::cout << 41 << std::endl;
            int ts = target_size;
            int ss = source_size;
            Matrix<CoefficientPrecision> H_dense(ts, size);
            std::cout << 42 << std::endl;

            Matrix<CoefficientPrecision> K_dense(size, source_size);
            std::cout << 43 << std::endl;

            std::cout << H->get_target_cluster().get_size() << ',' << H->get_source_cluster().get_size() << std::endl;
            std::cout << K->get_target_cluster().get_size() << ',' << K->get_source_cluster().get_size() << std::endl;
            std::cout << H_dense.nb_rows() << ',' << H_dense.nb_cols() << std::endl;
            std::cout << H->is_hierarchical() << ',' << H->is_dense() << ',' << H->is_low_rank() << std::endl;
            if (H->is_dense()) {
                std::cout << "alors ? " << std::endl;
                auto test = H->get_dense_data();
                std::cout << "acces ok" << std::endl;
                std::cout << test->nb_rows() << ',' << test->nb_cols() << std::endl;
            }
            copy_to_dense(*H, H_dense.data());
            std::cout << 44 << std::endl;

            copy_to_dense(*K, K_dense.data());
            std::cout << 45 << std::endl;

            // H_dense.add_matrix_product('N', 1.0, K_dense.data(), 1.0, res.data(), 1);
            std::cout << res.nb_rows() << ',' << res.nb_cols() << '=' << target_size << ',' << size << 'x' << size << ',' << source_size << std::endl;
            res = res + H_dense * K_dense;
            std::cout << "4 ok " << std::endl;
            // std::cout << " ca a été ok" << std::endl;
        }
        return res;
    }
};
} // namespace htool

#endif
