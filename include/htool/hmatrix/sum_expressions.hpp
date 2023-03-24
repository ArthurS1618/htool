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
class SumExpression : public VirtualGenerator<CoefficientPrecision> {
  private:
    using HMatrixType = HMatrix<CoefficientPrecision, CoordinatePrecision>;

    std::vector<const HMatrixType *> Sr;
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

    SumExpression(std::vector<const HMatrixType *> sr, std::vector<const HMatrixType *> sh, int target_size0, int target_offset0, int source_size0, int source_offset0) {
        Sr            = sr;
        Sh            = sh;
        target_size   = target_size0;
        target_offset = target_offset0;
        source_size   = source_size0;
        source_offset = source_offset0;
    }

    std::vector<const HMatrixType *> get_sr() { return Sr; }
    std::vector<const HMatrixType *> get_sh() { return Sh; }
    int get_nr() { return target_size; }
    int get_nc() { return source_size; }
    int get_target_offset() { return target_offset; }
    int get_source_offset() { return source_offset; }

    std::vector<CoefficientPrecision> prod(const std::vector<CoefficientPrecision> &x) {
        std::vector<CoefficientPrecision> y(target_size, 0.0);
        for (auto low_rank_leaf_hmatrix : Sr) {
            low_rank_leaf_hmatrix->add_vector_product('N', 1.0, x.data(), 1.0, y.data());
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

    CoefficientPrecision get_coeff(const int &i, const int &j) {
        std::vector<CoefficientPrecision> xj(source_size, 0.0);
        xj[j]                               = 1.0;
        std::vector<CoefficientPrecision> y = this->prod(xj);
        return y[i];
    }

    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        auto H = *this;
        for (int k = 0; k < M; ++k) {
            for (int l = 0; l < N; ++l) {
                std::vector<CoefficientPrecision> xl(source_size, 0.0);
                xl[l]                               = 1.0;
                std::vector<CoefficientPrecision> y = H.prod(xl);
                ptr[k + M * l]                      = y[k];
            }
        }
    }

    // ///////////////////////////////////////////////////////
    // // RESTRICT
    // /////////////////////////////////////////////

    SumExpression Restrict(int target_size0, int target_offset0, int source_size0, int source_offset0) const {
        std::vector<const HMatrixType *> sh;
        std::vector<const HMatrixType *> sr;
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
                                    if ((H_child->is_low_rank() and K_child->is_low_rank() or (H_child->is_low_rank() and !(K_child->is_low_rank())) or (!(H_child->is_low_rank()) and K_child->is_low_rank()))) {
                                        sr.push_back(H_child.get());
                                        sr.push_back(K_child.get());
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
                if (H->is_low_rank()) {
                    for (int l = 0; l < K_children.size(); ++l) {
                        auto &K_child = K_children[l];
                        if (K_child->get_source_cluster().get_size() == source_size and K_child->get_source_cluster().get_offset() == source_offset) {
                            sr.push_back(H);
                            sr.push_back(K_child.get());
                        }
                    }
                } else {
                    Matrix<CoefficientPrecision> H_dense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                    copy_to_dense(*H, H_dense.data());
                    auto &H_target_children = H->get_target_cluster().get_children();
                    int flag                = 0;
                    for (int k = 0; k < H_target_children.size(); ++k) {
                        auto &target_k = H_target_children[k];
                        if (target_k->get_size() == target_size0 and target_k->get_offset() == target_offset0) {
                            flag = k;
                        }
                    }
                    for (int l = 0; l < K_children.size(); ++l) {
                        auto &K_child = K_children[l];
                        if ((K_child->get_source_cluster().get_size() == source_size) and (K_child->get_source_cluster().get_offset() == source_offset0)) {
                            Matrix<CoefficientPrecision> H_restr(target_size0, K_child->get_target_cluster().get_offset());
                            CoefficientPrecision *ptr = new CoefficientPrecision[target_size * K_child->get_target_cluster().get_size()];
                            for (int j = 0; j < K_child->get_target_cluster().get_size(); ++j) {
                                for (int i = 0; i < target_size0; ++i) {
                                    ptr[i + j * target_size0] = H_dense(i + target_offset0 - H->get_target_cluster().get_offset(), j + K_child->get_target_cluster().get_offset() - H->get_source_cluster().get_offset());
                                }
                            }
                            H_restr.assign(target_size0, K_child->get_target_cluster().get_size(), ptr, true);
                            // auto &target_k                                               = *const_cast<std::unique_ptr<Cluster<CoordinatePrecision>> *>(&H_target_children[flag]);
                            // std::shared_ptr<Cluster<CoordinatePrecision>> target_cluster = std::make_shared<Cluster<CoordinatePrecision>>(*(target_k.get()));

                            // std::unique_ptr<Cluster<CoordinatePrecision>> target_cluster(target_k.release());
                            // std::unique_ptr<Cluster<CoordinatePrecision>> unique_target_cluster = std::move(target_cluster);
                            // std::unique_ptr<Cluster<CoordinatePrecision>> source_cluster(K_child->get_target_cluster());
                            auto &target    = H_target_children[flag];
                            auto target_ptr = std::shared_ptr<const htool::Cluster<double>>(target.get(), [](htool::Cluster<double> *p) {});
                            auto source_ptr = std::shared_ptr<const htool::Cluster<double>>(&K_child->get_target_cluster(), [](const htool::Cluster<double> *p) {});
                            HMatrixType H_k(target_ptr, source_ptr);
                            DenseGenerator<CoefficientPrecision> mat_H(H_restr);
                            H_k.compute_dense_data(mat_H);
                            sh.push_back(&H_k);
                            sh.push_back(K_child.get());
                        }
                    }
                }
            } else if (H_children.size() > 0 and K_children.size() == 0) {
                if (K->is_low_rank()) {
                    for (int k = 0; k < H_children.size(); ++k) {
                        auto &H_child = H_children[k];
                        if (H_child->get_target_cluster().get_size() == target_size0 and H_child->get_target_cluster().get_offset() == target_offset0) {
                            sr.push_back(H_child.get());
                            sr.push_back(K);
                        }
                    }
                } else {
                    Matrix<CoefficientPrecision> K_dense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                    copy_to_dense(*K, K_dense.data());
                    auto &K_source_children = K->get_target_cluster().get_children();
                    int flag                = 0;
                    for (int k = 0; k < K_source_children.size(); ++k) {
                        auto &source_k = K_source_children[k];
                        if ((source_k->get_size() == source_size0) and (source_k->get_offset() == source_offset0)) {
                            flag = k;
                        }
                    }
                    for (int l = 0; l < H_children.size(); ++l) {
                        auto &H_child = H_children[l];
                        if (H_child->get_target_cluster().get_size() == target_size0 and H_child->get_target_cluster().get_offset() == target_offset0) {
                            Matrix<CoefficientPrecision> K_restr(H_child->get_source_cluster().get_size(), source_size0);
                            CoefficientPrecision *ptr = new CoefficientPrecision[source_size0 * H_child->get_source_cluster().get_size()];
                            for (int j = 0; j < source_size0; ++j) {
                                for (int i = 0; i < H_child->get_source_cluster().get_size(); ++i) {
                                    ptr[i + j * H_child->get_source_cluster().get_size()] = K_dense(i + H->get_target_cluster().get_offset() - K->get_source_cluster().get_offset(), j + source_size0 - H->get_source_cluster().get_offset());
                                }
                            }
                            //                      // K_restr.assign(H_child->get_source_cluster().get_size(), source_size0, ptr, true);
                            //                      // std::shared_ptr<Cluster<CoordinatePrecision>> target_cluster = std::make_shared<Cluster<CoordinatePrecision>>(H_child->get_source_cluster());
                            //                      // std::shared_ptr<Cluster<CoordinatePrecision>> source_cluster = std::make_shared<Cluster<CoordinatePrecision>>(K_source_children[flag]);

                            auto &source    = K_source_children[flag];
                            auto source_ptr = std::shared_ptr<const htool::Cluster<CoordinatePrecision>>(source.get(), [](htool::Cluster<CoordinatePrecision> *p) {});
                            auto target_ptr = std::shared_ptr<const htool::Cluster<CoordinatePrecision>>(&H_child->get_source_cluster(), [](const htool::Cluster<CoordinatePrecision> *p) {});

                            HMatrixType K_k(target_ptr, source_ptr);
                            DenseGenerator<CoefficientPrecision> mat_K(K_restr);
                            K_k.compute_dense_data(mat_K);
                            sh.push_back(H_child.get());
                            sh.push_back(&K_k);
                        }
                    }
                }
            } else if ((H_children.size() == 0) and (K_children.size() == 0)) {
                if ((H->is_low_rank()) or (K->is_low_rank())) {
                    sr.push_back(H);
                    sr.push_back(K);
                } else {
                    Matrix<CoefficientPrecision> H_dense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                    Matrix<CoefficientPrecision> K_dense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                    copy_to_dense(*H, H_dense.data());
                    copy_to_dense(*K, K_dense.data());
                    Matrix<CoefficientPrecision> H_restr(target_size0, H->get_source_cluster().get_size());
                    Matrix<CoefficientPrecision> K_restr(K->get_target_cluster().get_size(), source_size0);
                    CoefficientPrecision *ptr_H = new CoefficientPrecision[target_size0 * H->get_source_cluster().get_size()];
                    CoefficientPrecision *ptr_K = new CoefficientPrecision[source_size0 * K->get_target_cluster().get_size()];
                    for (int j = 0; j < H->get_source_cluster().get_size(); ++j) {
                        for (int i = 0; i < target_size0; ++i) {
                            ptr_H[i + j * target_size0] = H_dense(i + target_offset - H->get_target_cluster().get_offset(), j);
                        }
                    }
                    for (int j = 0; j < source_size0; ++j) {
                        for (int i = 0; i < K->get_target_cluster().get_size(); ++i) {
                            ptr_K[i + j * K->get_target_cluster().get_size()] = K_dense(i + target_offset0 - H->get_target_cluster().get_offset(), j);
                        }
                    }
                    H_restr.assign(target_size0, H->get_source_cluster().get_size(), ptr_H, true);
                    K_restr.assign(K->get_target_cluster().get_size(), source_size0, ptr_K, true);
                    auto &target_children = H->get_target_cluster().get_children();
                    auto &source_children = K->get_source_cluster().get_children();
                    int flag_h            = 0;
                    int flag_k            = 0;
                    for (int k = 0; k < target_children.size(); ++k) {
                        auto &child = target_children[k];
                        if ((child->get_size() == target_size0) and (child->get_offset() == target_offset0)) {
                            flag_h = k;
                        }
                    }
                    for (int k = 0; k < source_children.size(); ++k) {
                        auto &child = source_children[k];
                        if ((child->get_size() == source_size0) and (child->get_offset() == source_offset0)) {
                            flag_h = k;
                        }
                    }
                    DenseGenerator<CoefficientPrecision> mat_H(H_restr);
                    DenseGenerator<CoefficientPrecision> mat_K(K_restr);
                    //   std::shared_ptr<Cluster<CoordinatePrecision>> target_cluster = std::make_shared<Cluster<CoordinatePrecision>>(target_children[flag_h]);
                    //   std::shared_ptr<Cluster<CoordinatePrecision>> source_cluster = std::make_shared<Cluster<CoordinatePrecision>>(source_children[flag_k]);
                    //   std::shared_ptr<Cluster<CoordinatePrecision>> mid_cluster    = std::make_shared<Cluster<CoordinatePrecision>>(H->get_source_cluster());

                    auto &source    = source_children[flag_k];
                    auto source_ptr = std::shared_ptr<const htool::Cluster<CoordinatePrecision>>(source.get(), [](htool::Cluster<CoordinatePrecision> *p) {});
                    auto &target    = target_children[flag_h];
                    auto target_ptr = std::shared_ptr<const htool::Cluster<CoordinatePrecision>>(target.get(), [](htool::Cluster<CoordinatePrecision> *p) {});
                    auto mid_ptr    = std::shared_ptr<const htool::Cluster<CoordinatePrecision>>(&H->get_source_cluster(), [](const htool::Cluster<CoordinatePrecision> *p) {});

                    HMatrixType H_k(target_ptr, mid_ptr);
                    H_k.compute_dense_data(mat_H);
                    HMatrixType K_k(mid_ptr, source_ptr);
                    H_k.compute_dense_data(mat_K);
                    sh.push_back(&H_k);
                    sh.push_back(&K_k);
                }
            }
        }
        SumExpression res(sr, sh, target_size0, target_offset0, source_size0, source_offset0);
        return res;
    }

    // /////////////////////////////////////////////////////
    // //// EVALUATE
    // //////////////////////////////////////////////

    Matrix<CoefficientPrecision> Evaluate() {
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
