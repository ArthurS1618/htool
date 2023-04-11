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

    std::vector<const Matrix<CoefficientPrecision> *> Sr;
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

    SumExpression(const std::vector<const Matrix<CoefficientPrecision> *> &sr, const std::vector<const HMatrixType *> &sh, const std::vector<int> offset0, const int target_size0, const int target_offset0, const int source_size0, const int source_offset0) {
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

    const std::vector<const Matrix<CoefficientPrecision> *> get_sr() const { return Sr; }
    const std::vector<const HMatrixType *> get_sh() const { return Sh; }
    const int get_nr() { return target_size; }
    const int get_nc() { return source_size; }
    const int get_target_offset() { return target_offset; }
    const int get_source_offset() { return source_offset; }
    const std::vector<int> get_offset() { return offset; }

    std::vector<CoefficientPrecision> prod(const std::vector<CoefficientPrecision> &x) const {
        std::vector<CoefficientPrecision> y(target_size, 0.0);
        for (int k = 0; k < Sr.size() / 2; ++k) {
            const Matrix<CoefficientPrecision> *U = Sr[2 * k];
            const Matrix<CoefficientPrecision> *V = Sr[2 * k + 1];
            int oft                               = offset[2 * k];
            int ofs                               = offset[2 * k + 1];
            auto uv                               = *U * (*V);
            CoefficientPrecision *ptr             = new CoefficientPrecision[target_size * source_size];
            for (int i = 0; i < target_size; ++i) {
                for (int j = 0; j < source_size; ++j) {
                    ptr[i + target_size * j] = uv(i + target_offset - oft, j + source_offset - ofs);
                }
            }
            Matrix<CoefficientPrecision> uvrestr(target_size, source_size);
            uvrestr.assign(target_size, source_size, ptr, true);
            y = y + uvrestr * x;
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

    CoefficientPrecision get_coeff(const int &i, const int &j) const {
        std::vector<CoefficientPrecision> xj(source_size, 0.0);
        xj[j]                               = 1.0;
        std::vector<CoefficientPrecision> y = this->prod(xj);
        return y[i];
    }

    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        for (int k = 0; k < M; ++k) {
            for (int l = 0; l < N; ++l) {
                ptr[k + M * l] = this->get_coeff(k + row_offset, l + col_offset);
            }
        }
    }

    // void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
    //     Matrix<CoefficientPrecision> temp(M, N);
    //     for (int k = 0; k < Sr.size(); ++k) {
    //         auto &lr = Sr[k];
    //         int oft  = offset[2 * k];
    //         int ofs  = offset[2 * k + 1];
    //         // il faut déja restreindre la lr au bon bloc et ensuite en faire la restr
    //         CoefficientPrecision *ptru = new CoefficientPrecision[M * lr->rank_of()];
    //         CoefficientPrecision *ptrv = new CoefficientPrecision[lr->rank_of() * N];
    //         for (int i = 0; i < M; ++i) {
    //             for (int j = 0; j < lr->rank_of(); ++j) {
    //                 ptru[i + j * M] = lr->get_U(i + (target_offset - oft) + row_offset, j);
    //             }
    //         }
    //         for (int i = 0; i < lr->rank_of(); ++i) {
    //             for (int j = 0; j < N; ++j) {
    //                 ptrv[i + lr->rank_of() * j] = lr->get_V(i, j + col_offset + (source_offset - ofs));
    //             }
    //         }
    //         Matrix<CoefficientPrecision> U(M, lr->rank_of());
    //         Matrix<CoefficientPrecision> V(lr->rank_of(), N);
    //         U.assign(M, lr->rank_of(), ptru, true);
    //         V.assign(lr->rank_of(), N, ptrv, true);
    //         temp = temp + U * V;
    //     }
    //     for (int k = 0; Sh.size() / 2; ++k) {
    //         auto &H = Sh[2 * k];
    //         auto &K = Sh[2 * k + 1];
    //         for (int j = 0; j < N; ++j) {
    //             std::vector<CoefficientPrecision> ej(K->get_source_cluster().get_size(), 0.0);
    //             ej[j + col_offset] = 1;
    //             std::vector<CoefficientPrecision> ytemp(K->get_target_cluster().get_size(), 0.0);
    //             std::vector<CoefficientPrecision> col_j(H->get_target_cluster().get_size(), 0.0);
    //             K->add_vector_product('N', 1.0, ej.data(), 0.0, ytemp.data());
    //             H->add_vector_product('N', 1.0, ytemp.data(), 0.0, col_j.data());
    //             for (int i = 0; i < N; ++i) {
    //                 temp(i, j) = temp(i, j) + col_j[i + row_offset];
    //             }
    //         }
    //     }
    //     std::copy_n(temp.data(), M * N, ptr);
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

    // ///////////////////////////////////////////////////////
    // // RESTRICT
    // /////////////////////////////////////////////

    // On est censé appeler restrict (tau , sigma) seulementsi il y a pas de dense plus grande que tau a gauche ou de dense plus grande que sigma a droite
    const SumExpression Restrict(int target_size0, int target_offset0, int source_size0, int source_offset0) const {
        std::vector<const HMatrixType *> sh;
        std::vector<const Matrix<CoefficientPrecision> *> sr;
        auto of = offset;
        for (auto s : Sr) {
            sr.push_back(s);
        }
        std::cout << "au debut" << std::endl;
        if (Sr.size() > 0) {
            std::cout << Sr[0]->nb_rows() << ',' << Sr[0]->nb_cols() << ',' << Sr[1]->nb_rows() << ',' << Sr[1]->nb_cols() << std::endl;
        }
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
                                            std::cout << "!1" << std::endl;
                                            Matrix<CoefficientPrecision> uh = H_child->get_low_rank_data()->Get_U();
                                            Matrix<CoefficientPrecision> vh = H_child->get_low_rank_data()->Get_V();
                                            Matrix<CoefficientPrecision> uk = K_child->get_low_rank_data()->Get_U();
                                            Matrix<CoefficientPrecision> vk = K_child->get_low_rank_data()->Get_V();
                                            Matrix<CoefficientPrecision> v  = vh * uk * vk;
                                            std::cout << "on push " << uh.nb_rows() << ',' << uh.nb_cols() << ',' << v.nb_rows() << ',' << v.nb_cols() << std::endl;

                                            sr.push_back(&uh);
                                            sr.push_back(&v);
                                            std::cout << "1 :" << sr[0]->nb_rows() << ',' << sr[0]->nb_cols() << ',' << sr[1]->nb_rows() << ',' << sr[1]->nb_cols() << std::endl;

                                            of.push_back(H_child->get_target_cluster().get_offset());
                                            of.push_back(K->get_source_cluster().get_offset());
                                        } else if ((H_child->is_low_rank()) and !(K_child->is_low_rank())) {
                                            std::cout << "!2" << std::endl;
                                            Matrix<CoefficientPrecision> u  = H_child->get_low_rank_data()->Get_U();
                                            Matrix<CoefficientPrecision> v  = H_child->get_low_rank_data()->Get_V();
                                            Matrix<CoefficientPrecision> vk = K_child->mult(v, 'T');
                                            std::cout << "size avant " << sr.size() << std::endl;
                                            if (sr.size() > 0) {
                                                std::cout << "2 avant:" << sr[0]->nb_rows() << ',' << sr[0]->nb_cols() << ',' << sr[1]->nb_rows() << ',' << sr[1]->nb_cols() << std::endl;
                                            }
                                            std::cout << "on push " << u.nb_rows() << ',' << u.nb_cols() << ',' << vk.nb_rows() << ',' << vk.nb_cols() << std::endl;
                                            sr.push_back(&u);
                                            sr.push_back(&vk);
                                            std::cout << "2 apres :" << sr[0]->nb_rows() << ',' << sr[0]->nb_cols() << ',' << sr[1]->nb_rows() << ',' << sr[1]->nb_cols() << std::endl;

                                            std::cout << "size apres " << sr.size() << std::endl;
                                            of.push_back(H_child->get_target_cluster().get_offset());
                                            of.push_back(K_child->get_source_cluster().get_offset());
                                        } else if (!(H_child->is_low_rank()) and (K_child->is_low_rank())) {
                                            std::cout << "!3" << std::endl;
                                            Matrix<CoefficientPrecision> u  = K_child->get_low_rank_data()->Get_U();
                                            Matrix<CoefficientPrecision> v  = K_child->get_low_rank_data()->Get_V();
                                            Matrix<CoefficientPrecision> hu = H_child->mult(u, 'N');
                                            std::cout << hu.nb_rows() << ',' << hu.nb_cols() << ',' << v.nb_rows() << ',' << v.nb_cols() << std::endl;
                                            sr.push_back(&hu);
                                            sr.push_back(&v);
                                            std::cout << "3:" << sr[0]->nb_rows() << ',' << sr[0]->nb_cols() << ',' << sr[1]->nb_rows() << ',' << sr[1]->nb_cols() << std::endl;
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
                    Matrix<CoefficientPrecision> H_dense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                    copy_to_dense(*H, H_dense.data());
                    // donc la normalement K a juste 2 fils et on prend celui qui a le bon sigma
                    for (int l = 0; l < K_children.size(); ++l) {
                        auto &K_child = K_children[l];
                        if ((K_child->get_source_cluster().get_size() == source_size0) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
                            // soit k_child est lr soit il l'est pas ;
                            if (K_child->is_low_rank()) {
                                std::cout << "!4" << std::endl;
                                Matrix<CoefficientPrecision> u  = K_child->get_low_rank_data()->Get_U();
                                Matrix<CoefficientPrecision> v  = K_child->get_low_rank_data()->Get_V();
                                Matrix<CoefficientPrecision> Hu = H_dense * u;
                                sr.push_back(&Hu);
                                sr.push_back(&v);

                                std::cout << "4 :" << sr[0]->nb_rows() << ',' << sr[0]->nb_cols() << ',' << sr[1]->nb_rows() << ',' << sr[1]->nb_cols() << std::endl;

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
                    Matrix<CoefficientPrecision> K_dense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                    copy_to_dense(*K, K_dense.data());
                    for (int l = 0; l < H_children.size(); ++l) {
                        auto &H_child = H_children[l];
                        if (H_child->get_target_cluster().get_size() == target_size0 and H_child->get_target_cluster().get_offset() == target_offset0) {
                            if (H_child->is_low_rank()) {
                                std::cout << "!5" << std::endl;
                                Matrix<CoefficientPrecision> u  = H_child->get_low_rank_data()->Get_U();
                                Matrix<CoefficientPrecision> v  = H_child->get_low_rank_data()->Get_V();
                                Matrix<CoefficientPrecision> vK = v * K_dense;
                                sr.push_back(&u);
                                sr.push_back(&vK);
                                std::cout << "5 :" << sr[0]->nb_rows() << ',' << sr[0]->nb_cols() << ',' << sr[1]->nb_rows() << ',' << sr[1]->nb_cols() << std::endl;

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
        std::cout << "a la fin" << std::endl;
        if (sr.size() > 0) {
            std::cout << sr[0]->nb_rows() << ',' << sr[0]->nb_cols() << ',' << sr[1]->nb_rows() << ',' << sr[1]->nb_cols() << std::endl;
        }
        SumExpression res(sr, sh, of, target_size0, target_offset0, source_size0, source_offset0);
        if (sr.size() > 0) {
            std::cout << "test" << res.get_sr()[0]->nb_rows() << ',' << res.get_sr()[0]->nb_cols() << ',' << res.get_sr()[1]->nb_rows() << ',' << res.get_sr()[1]->nb_cols() << std::endl;
        }
        return res;
    }
};
} // namespace htool

#endif
