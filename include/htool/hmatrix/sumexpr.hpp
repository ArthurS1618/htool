#ifndef SUMEXPR_HPP
#define SUMEXPR_HPP
#include "../basic_types/vector.hpp"
#include <random>

namespace htool {
// a améliorer même si on est pas censé appelé ca
template <typename T>
Matrix<T> copy_sub_matrix(const Matrix<T> in, int nr, int nc, int oft, int ofs) {
    Matrix<T> res(nr, nc);
    for (int k = 0; k < nr; ++k) {
        for (int l = 0; l < nc; ++l) {
            res(k, l) = in(k + oft, l + ofs);
        }
    }
    return res;
}
template <typename CoefficientPrecision, typename CoordinatePrecision>
class SumExpression_fast : public VirtualGenerator<CoefficientPrecision> {
  private:
    using HMatrixType = HMatrix<CoefficientPrecision, CoordinatePrecision>;
    using LRtype      = LowRankMatrix<CoefficientPrecision, CoordinatePrecision>;

    // one to rule theim all
    std::vector<Matrix<CoefficientPrecision>> Sr;
    std::vector<HMatrixType *> Sh;       //// sum (HK)
    Matrix<CoefficientPrecision> Sdense; // pour gérer les petites denses
    int target_size;
    int target_offset;
    int source_size;
    int source_offset;
    bool is_restrictible;

  public:
    SumExpression_fast()
        : target_size(0), target_offset(0), source_size(0), source_offset(0) {
    }
    // SumExpression_fast() {
    // }
    SumExpression_fast(HMatrixType *A, HMatrixType *B) {
        Sh.push_back(A);
        Sh.push_back(B);
        target_size     = A->get_target_cluster().get_size();
        target_offset   = A->get_target_cluster().get_offset();
        source_size     = B->get_source_cluster().get_size();
        source_offset   = B->get_source_cluster().get_offset();
        is_restrictible = ((A->get_children().size() > 0) && (B->get_children().size()) > 0);
    }

    SumExpression_fast(const HMatrixType *A, const HMatrixType *B) {
        Sh.push_back(const_cast<HMatrixType *>(A));
        Sh.push_back(const_cast<HMatrixType *>(B));
        target_size     = A->get_target_cluster().get_size();
        target_offset   = A->get_target_cluster().get_offset();
        source_size     = B->get_source_cluster().get_size();
        source_offset   = B->get_source_cluster().get_offset();
        is_restrictible = ((A->get_children().size() > 0) && (B->get_children().size()) > 0);
    }

    SumExpression_fast(const std::vector<Matrix<CoefficientPrecision>> &sr, const std::vector<const HMatrixType *> &sh, const int &target_size0, const int &source_size0, const int &target_offset0, const int &source_offset0) {
        Sr            = sr;
        Sh            = sh;
        target_size   = target_size0;
        target_offset = target_offset0;
        source_size   = source_size0;
        source_offset = source_offset0;
    }
    SumExpression_fast<CoefficientPrecision, CoordinatePrecision> &operator=(const SumExpression_fast<CoefficientPrecision, CoordinatePrecision> &other) {
        if (this != &other) { // Vérification de l'auto-affectation
            Sr            = other.Sr;
            Sh            = other.Sh;
            target_size   = other.target_size;
            target_offset = other.target_offset;
            source_size   = other.source_size;
            source_offset = other.source_offset;
        }
        return *this;
    }
    ////////////////////////////
    //// Getter
    const std::vector<Matrix<CoefficientPrecision>> get_sr() const { return Sr; }
    const std::vector<HMatrixType *> get_sh() const { return Sh; }
    int get_nr() const { return target_size; }
    int get_nc() const { return source_size; }
    int get_target_offset() const { return target_offset; }
    int get_source_offset() const { return source_offset; }
    int get_target_size() const { return target_size; }
    int get_source_size() const { return source_size; }

    /////////////////////////
    //// Setter
    void set_dense(const Matrix<CoefficientPrecision> &sdense) {
        Sdense = sdense;
    }
    void set_restr(bool &flag) {
        is_restrictible = flag;
    }
    void set_sr(std::vector<Matrix<CoefficientPrecision>> &sr, const int &target_0, const int &source_0) {
        Sr            = sr;
        target_offset = target_0;
        source_offset = source_0;
        target_size   = sr[0].nb_rows();
        source_size   = sr[1].nb_cols();
    }

    void set_sh(std::vector<HMatrixType *> &sh0) { Sh = sh0; }
    void set_size(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) {
        target_offset = t.get_offset();
        source_offset = s.get_offset();
        target_size   = t.get_size();
        source_size   = s.get_size();
    }

    bool get_restr() const { return is_restrictible; }
    /////////////////////////////////////////////////////////////////////////
    /// PRODUCT SUMEXPR vect
    ////////////////////////////////////////////////////////////

    std::vector<CoefficientPrecision> prod(const char T, const std::vector<CoefficientPrecision> &x) const {
        if (T == 'N') {
            std::vector<CoefficientPrecision> y(target_size, 0.0);
            if (Sr.size() > 0) {
                for (int k = 0; k < Sr.size() / 2; ++k) {
                    auto u = Sr[2 * k];
                    auto v = Sr[2 * k + 1];
                    std::vector<CoefficientPrecision> ytemp(v.nb_rows(), 0.0);
                    v.add_vector_product('N', 1.0, x.data(), 0.0, ytemp.data());
                    u.add_vector_product('N', 1.0, ytemp.data(), 1.0, y.data());
                }
            }

            for (int k = 0; k < Sh.size() / 2; ++k) {
                auto H = Sh[2 * k];
                auto K = Sh[2 * k + 1];
                std::vector<CoefficientPrecision> y_temp(K->get_target_cluster().get_size(), 0.0);
                K->add_vector_product('N', 1.0, x.data(), 0.0, y_temp.data());
                H->add_vector_product('N', 1.0, y_temp.data(), 1.0, y.data());
            }
            if (Sdense.nb_cols() > 0) {
                Sdense.add_vector_product('N', 1.0, x.data(), 1.0, y.data());
            }
            return (y);
        }

        else {
            std::vector<CoefficientPrecision> y(source_size, 0.0);
            if (Sr.size() > 0) {
                for (int k = 0; k < Sr.size() / 2; ++k) {

                    auto u = Sr[2 * k];
                    auto v = Sr[2 * k + 1];
                    std::vector<CoefficientPrecision> ytemp(u.nb_cols(), 0.0);

                    u.add_vector_product('T', 1.0, x.data(), 0.0, ytemp.data());
                    v.add_vector_product('T', 1.0, ytemp.data(), 1.0, y.data());
                }
            }

            // y = y +(x* urestr) * vrestrx;

            for (int k = 0; k < Sh.size() / 2; ++k) {
                auto H = Sh[2 * k];
                auto K = Sh[2 * k + 1];
                std::vector<CoefficientPrecision> y_temp(H->get_source_cluster().get_size(), 0.0);
                H->add_vector_product('T', 1.0, x.data(), 0.0, y_temp.data());
                K->add_vector_product('T', 1.0, y_temp.data(), 1.0, y.data());
            }
            if (Sdense.nb_cols() > 0) {
                Sdense.add_vector_product('T', 1.0, x.data(), 1.0, y.data());
            }
            return (y);
        }
    }

    ////////////
    /// get _row & get_col by performing Sum_expr vector multiplication
    std::vector<CoefficientPrecision> get_col(const int &l) const {
        std::vector<double> e_l(source_size);
        e_l[l]   = 1.0;
        auto col = this->prod('N', e_l);
        return col;
    }
    std::vector<CoefficientPrecision> get_row(const int &k) const {
        std::vector<double> e_k(target_size);
        e_k[k]   = 1.0;
        auto row = this->prod('T', e_k);
        return row;
    }
    //////////////////////////////////////////
    //// COp submatrix
    //////////////////////////////////////
    // double norme_approx() {
    //     double nr = 0.0;
    // }
    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        // on regarde si il faut pas juste une ligne ou une collonne
        Matrix<CoefficientPrecision> mat(M, N);
        mat.assign(M, N, ptr, false);
        int ref_row = row_offset - target_offset;
        int ref_col = col_offset - source_offset;
        if (M == 1) {
            auto row = this->get_row(ref_row);
            std::copy(row.begin() + ref_col, row.begin() + ref_col + N, mat.data());
        } else if (N == 1) {
            auto col = this->get_col(ref_col);
            std::copy(col.begin() + ref_row, col.begin() + ref_row + M, mat.data());
        }

        else {
            // on veut toute la matrice
            for (int k = 0; k < Sh.size() / 2; ++k) {
                auto &H = Sh[2 * k];
                auto &K = Sh[2 * k + 1];
                if (H->get_target_cluster().get_size() != target_size || K->get_source_cluster().get_size() != source_size) {
                    std::cout << "Cas pas normal : H et K pas allignés" << std::endl;
                }
                Matrix<CoefficientPrecision> hdense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                Matrix<CoefficientPrecision> kdense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                copy_to_dense(*H, hdense.data());
                copy_to_dense(*K, kdense.data());
                auto hrestr = copy_sub_matrix(hdense, M, hdense.nb_cols(), row_offset - target_offset, 0);
                auto krestr = copy_sub_matrix(kdense, kdense.nb_rows(), N, 0, col_offset - source_offset);
                hrestr.add_matrix_product('N', 1.0, krestr.data(), 1.0, mat.data(), N);
                // mat = mat + hrestr * krestr;
            }
            if (Sr.size() > 0) {
                for (int k = 0; k < Sr.size() / 2; ++k) {
                    auto U = Sr[2 * k];
                    auto V = Sr[2 * k + 1];
                    Matrix<CoefficientPrecision> U_restr(target_size, U.nb_cols());
                    Matrix<CoefficientPrecision> V_restr(V.nb_rows(), source_size);
                    for (int i = 0; i < target_size; ++i) {
                        U_restr.set_row(i, U.get_row(i + row_offset - target_offset));
                    }
                    for (int j = 0; j < source_size; ++j) {
                        V_restr.set_col(j, V.get_col(j + col_offset - source_offset));
                    }
                    U_restr.add_matrix_product('N', 1.0, V_restr.data(), 1.0, mat.data(), N);
                    // mat = mat + U_restr * V_restr;
                }
            }
            if (Sdense.nb_cols() > 0) {
                mat.plus_egal(Sdense);
                // mat = mat + copy_sub_matrix(Sdense, M, N, ref_row, ref_col);
            }
        }
        std::copy(mat.data(), mat.data() + M * N, ptr);
    }

    // do Sr = Sr +AB with QR svd
    // U1V1^T+U2V2^T = (U1, U2)(V1,V2)^T = (Q1R1)(Q2R2)T =

    SumExpression_fast restrict_ACA(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) const {
        bool flag = true;
        std::vector<Matrix<CoefficientPrecision>> sr_0;
        Matrix<CoefficientPrecision> sdense_0;
        // truncation SR
        if (Sr.size() > 0) {
            Matrix<CoefficientPrecision> U = Sr[0];
            Matrix<CoefficientPrecision> V = Sr[1];
            Matrix<CoefficientPrecision> Urestr(t.get_size(), U.nb_cols());
            Matrix<CoefficientPrecision> Vrestr(V.nb_rows(), s.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                Urestr.set_row(k, U.get_row(k + t.get_offset() - target_offset));
            }
            for (int l = 0; l < s.get_size(); ++l) {
                Vrestr.set_col(l, V.get_col(l + s.get_offset() - source_offset));
            }
            std::vector<Matrix<CoefficientPrecision>> temp(2);
            temp[0] = Urestr;
            temp[1] = Vrestr;

            sr_0 = temp;
        }
        if (Sdense.nb_cols() > 0) {
            sdense_0 = copy_sub_matrix(Sdense, t.get_size(), s.get_size(), t.get_offset() - target_offset, s.get_offset() - source_offset);
        }
        std::vector<HMatrixType *> sh_0;
        SumExpression_fast res;
        // On parcours Sh si un des deux admissible on fait un ACA avec une sumexpr temporaire qui a deux matrices lw rank SR+ HK (H ou K lr)
        for (int k = 0; k < Sh.size() / 2; ++k) {
            auto &H = Sh[2 * k];
            auto &K = Sh[2 * k + 1];
            for (auto &rho_child : H->get_source_cluster().get_children()) {
                bool test_target = H->compute_admissibility(t, *rho_child);
                bool test_source = K->compute_admissibility(*rho_child, s);

                auto H_child = H->get_block(t.get_size(), rho_child->get_size(), t.get_offset(), rho_child->get_offset());
                auto K_child = K->get_block(rho_child->get_size(), s.get_size(), rho_child->get_offset(), s.get_offset());
                // si on arrive pas a choper les blocs : ca doit pas arriver mais bon
                if (H_child->get_target_cluster().get_size() != t.get_size() || H_child->get_source_cluster().get_size() != rho_child->get_size() || K_child->get_target_cluster().get_size() != rho_child->get_size() || K_child->get_source_cluster().get_size() != s.get_size()) {
                    // std::cout << "?! " << std::endl;
                    // std::cout << H_child->get_target_cluster().get_size() << '=' << t.get_size() << ',' << H_child->get_source_cluster().get_size() << '=' << rho_child->get_size() << ',' << K_child->get_target_cluster().get_size() << '=' << rho_child->get_size() << ',' << K_child->get_source_cluster().get_size() << '=' << s.get_size() << std::endl;
                    Matrix<CoefficientPrecision> hdense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                    Matrix<CoefficientPrecision> kdense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                    copy_to_dense(*H, hdense.data());
                    copy_to_dense(*K, kdense.data());
                    std::cout << "            avant d'extraire : " << normFrob(hdense) << ',' << normFrob(kdense) << std::endl;
                    auto hrestr                = copy_sub_matrix(hdense, t.get_size(), rho_child->get_size(), t.get_offset() - H->get_target_cluster().get_offset(), rho_child->get_offset() - H->get_source_cluster().get_offset());
                    auto krestr                = copy_sub_matrix(kdense, rho_child->get_size(), s.get_size(), rho_child->get_offset() - K->get_target_cluster().get_offset(), s.get_offset() - K->get_source_cluster().get_offset());
                    const char trans           = 'N';
                    int m                      = t.get_size();
                    int n                      = s.get_size();
                    int kk                     = rho_child->get_size();
                    CoefficientPrecision alpha = 1.0;
                    CoefficientPrecision beta  = 1.0;
                    int lda                    = m;
                    int ldb                    = kk;
                    int ldc                    = m;
                    if (sdense_0.nb_cols() == 0) {
                        sdense_0.resize(t.get_size(), s.get_size());
                    }
                    Blas<CoefficientPrecision>::gemm(&trans, &trans, &m, &n, &kk, &alpha, hrestr.data(), &lda, krestr.data(), &ldb, &beta, sdense_0.data(), &ldc);
                    std::cout << "sdense de norme : " << normFrob(sdense_0) << ',' << sdense_0.nb_rows() << ',' << sdense_0.nb_cols() << std::endl;
                    flag = false;

                } else {
                    // sinon on regarde si ils sont admissibles
                    if ((H_child->is_low_rank() || K_child->is_low_rank() || test_target || test_source) && (t.get_children().size() > 0) && (s.get_children().size() > 0)) {
                        // Compute low-rank matrix UV =  LR(HK) ;
                        SumExpression_fast temp(H_child, K_child);
                        if (sr_0.size() == 0) {
                            // Première affectation
                            LRtype lr_new(temp, *H_child->get_low_rank_generator(), t, s, -1, H_child->get_epsilon());
                            if (lr_new.get_U().nb_cols() > 0) {
                                // ACA a marché
                                std::vector<Matrix<CoefficientPrecision>> tempm(2);
                                tempm[0] = lr_new.get_U();
                                tempm[1] = lr_new.get_V();
                                sr_0     = tempm;

                            } else {
                                // on push les hmat
                                if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
                                    flag = false;
                                }
                                sh_0.push_back(H_child);
                                sh_0.push_back(K_child);
                            }
                        } else {
                            // il faut rajouter HK a SR, on appel fait un ACA sur SUMEXPR(HK, SR)
                            SumExpression_fast s_intermediaire(H_child, K_child);
                            std::vector<Matrix<CoefficientPrecision>> data_lr(2);
                            data_lr[0] = sr_0[0];
                            data_lr[1] = sr_0[1];
                            s_intermediaire.set_sr(data_lr, t.get_offset(), s.get_offset());
                            LRtype lr_bourrin(s_intermediaire, *H_child->get_low_rank_generator(), t, s, -1, H_child->get_epsilon());
                            if (lr_bourrin.get_U().nb_cols() > 0) {
                                // ACA a marché
                                std::vector<Matrix<CoefficientPrecision>> tempm(2);
                                tempm[0] = lr_bourrin.get_U();
                                tempm[1] = lr_bourrin.get_V();
                                sr_0     = tempm;

                            } else {
                                // on push les matrice hiérachique
                                if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
                                    flag = false;
                                }
                                sh_0.push_back(H_child);
                                sh_0.push_back(K_child);
                            }
                        }

                    } else {
                        // Pas admissble , on push les hmat
                        if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
                            flag = false;
                        }
                        sh_0.push_back(H_child);
                        sh_0.push_back(K_child);
                    }
                }
            }
        }
        if (sr_0.size() > 0) {
            int oft = t.get_offset();
            int ofs = s.get_offset();
            res.set_sr(sr_0, oft, ofs);
        }
        if (sh_0.size() > 0) {
            res.set_sh(sh_0);
        }
        res.set_size(t, s);
        res.set_restr(flag);
        res.set_dense(sdense_0);

        return res;
    }

    SumExpression_fast restrict_ACA_triangulaire(char transa, char transb, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) const {
        bool flag = true;
        std::vector<Matrix<CoefficientPrecision>> sr_0;
        Matrix<CoefficientPrecision> sdense_0;
        // truncation SR
        if (Sr.size() > 0) {
            Matrix<CoefficientPrecision> U = Sr[0];
            Matrix<CoefficientPrecision> V = Sr[1];
            Matrix<CoefficientPrecision> Urestr(t.get_size(), U.nb_cols());
            Matrix<CoefficientPrecision> Vrestr(V.nb_rows(), s.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                Urestr.set_row(k, U.get_row(k + t.get_offset() - target_offset));
            }
            for (int l = 0; l < s.get_size(); ++l) {
                Vrestr.set_col(l, V.get_col(l + s.get_offset() - source_offset));
            }
            std::vector<Matrix<CoefficientPrecision>> temp(2);
            temp[0] = Urestr;
            temp[1] = Vrestr;

            sr_0 = temp;
        }
        if (Sdense.nb_cols() > 0) {
            sdense_0 = copy_sub_matrix(Sdense, t.get_size(), s.get_size(), t.get_offset() - target_offset, s.get_offset() - source_offset);
        }
        std::vector<HMatrixType *> sh_0;
        SumExpression_fast res;
        // On parcours Sh si un des deux admissible on fait un ACA avec une sumexpr temporaire qui a deux matrices lw rank SR+ HK (H ou K lr)
        for (int k = 0; k < Sh.size() / 2; ++k) {
            auto &H = Sh[2 * k];
            auto &K = Sh[2 * k + 1];
            for (auto &rho_child : H->get_source_cluster().get_children()) {
                if (transa == 'L') {
                    if (rho_child->get_offset() > t.get_offset()) {
                        continue;
                    }
                } else if (transa == 'U') {
                    if (rho_child->get_offset() < t.get_offset()) {
                        continue;
                    }
                }
                if (transb == 'L') {
                    if (rho_child->get_offset() < s.get_offset()) {
                        continue;
                    }
                } else if (transb == 'U') {
                    if (rho_child->get_offset() > s.get_offset()) {
                        continue;
                    }
                }
                bool test_target = H->compute_admissibility(t, *rho_child);
                bool test_source = H->compute_admissibility(*rho_child, s);

                auto H_child = H->get_block(t.get_size(), rho_child->get_size(), t.get_offset(), rho_child->get_offset());
                auto K_child = K->get_block(rho_child->get_size(), s.get_size(), rho_child->get_offset(), s.get_offset());
                // si on arrive pas a choper les blocs : ca doit pas arriver mais bon
                if (H_child->get_target_cluster().get_size() != t.get_size() || H_child->get_source_cluster().get_size() != rho_child->get_size() || K_child->get_target_cluster().get_size() != rho_child->get_size() || K_child->get_source_cluster().get_size() != s.get_size()) {
                    // std::cout << "?! " << std::endl;
                    // std::cout << H_child->get_target_cluster().get_size() << '=' << t.get_size() << ',' << H_child->get_source_cluster().get_size() << '=' << rho_child->get_size() << ',' << K_child->get_target_cluster().get_size() << '=' << rho_child->get_size() << ',' << K_child->get_source_cluster().get_size() << '=' << s.get_size() << std::endl;
                    Matrix<CoefficientPrecision> hdense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                    Matrix<CoefficientPrecision> kdense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                    copy_to_dense(*H, hdense.data());
                    copy_to_dense(*K, kdense.data());
                    std::cout << "            avant d'extraire : " << normFrob(hdense) << ',' << normFrob(kdense) << std::endl;
                    auto hrestr                = copy_sub_matrix(hdense, t.get_size(), rho_child->get_size(), t.get_offset() - H->get_target_cluster().get_offset(), rho_child->get_offset() - H->get_source_cluster().get_offset());
                    auto krestr                = copy_sub_matrix(kdense, rho_child->get_size(), s.get_size(), rho_child->get_offset() - K->get_target_cluster().get_offset(), s.get_offset() - K->get_source_cluster().get_offset());
                    const char trans           = 'N';
                    int m                      = t.get_size();
                    int n                      = s.get_size();
                    int kk                     = rho_child->get_size();
                    CoefficientPrecision alpha = 1.0;
                    CoefficientPrecision beta  = 1.0;
                    int lda                    = m;
                    int ldb                    = kk;
                    int ldc                    = m;
                    if (sdense_0.nb_cols() == 0) {
                        sdense_0.resize(t.get_size(), s.get_size());
                    }
                    Blas<CoefficientPrecision>::gemm(&trans, &trans, &m, &n, &kk, &alpha, hrestr.data(), &lda, krestr.data(), &ldb, &beta, sdense_0.data(), &ldc);
                    std::cout << "sdense de norme : " << normFrob(sdense_0) << ',' << sdense_0.nb_rows() << ',' << sdense_0.nb_cols() << std::endl;
                    flag = false;

                } else {
                    // sinon on regarde si ils sont admissibles
                    if ((H_child->is_low_rank() || K_child->is_low_rank() || test_target || test_source) && (t.get_children().size() > 0) && (s.get_children().size() > 0)) {
                        // Compute low-rank matrix UV =  LR(HK) ;
                        SumExpression_fast temp(H_child, K_child);
                        if (sr_0.size() == 0) {
                            // Première affectation
                            LRtype lr_new(temp, *H_child->get_low_rank_generator(), t, s, -1, H_child->get_epsilon());
                            if (lr_new.get_U().nb_cols() > 0) {
                                // ACA a marché
                                std::vector<Matrix<CoefficientPrecision>> tempm(2);
                                tempm[0] = lr_new.get_U();
                                tempm[1] = lr_new.get_V();
                                sr_0     = tempm;

                            } else {
                                // on push les hmat
                                if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
                                    flag = false;
                                }
                                sh_0.push_back(H_child);
                                sh_0.push_back(K_child);
                            }
                        } else {
                            // il faut rajouter HK a SR, on appel fait un ACA sur SUMEXPR(HK, SR)
                            SumExpression_fast s_intermediaire(H_child, K_child);
                            std::vector<Matrix<CoefficientPrecision>> data_lr(2);
                            data_lr[0] = sr_0[0];
                            data_lr[1] = sr_0[1];
                            s_intermediaire.set_sr(data_lr, t.get_offset(), s.get_offset());
                            LRtype lr_bourrin(s_intermediaire, *H_child->get_low_rank_generator(), t, s, -1, H_child->get_epsilon());
                            if (lr_bourrin.get_U().nb_cols() > 0) {
                                // ACA a marché
                                std::vector<Matrix<CoefficientPrecision>> tempm(2);
                                tempm[0] = lr_bourrin.get_U();
                                tempm[1] = lr_bourrin.get_V();
                                sr_0     = tempm;

                            } else {
                                // on push les matrice hiérachique
                                if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
                                    flag = false;
                                }
                                sh_0.push_back(H_child);
                                sh_0.push_back(K_child);
                            }
                        }

                    } else {
                        // Pas admissble , on push les hmat
                        if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
                            flag = false;
                        }
                        sh_0.push_back(H_child);
                        sh_0.push_back(K_child);
                    }
                }
            }
        }
        if (sr_0.size() > 0) {
            int oft = t.get_offset();
            int ofs = s.get_offset();
            res.set_sr(sr_0, oft, ofs);
        }
        if (sh_0.size() > 0) {
            res.set_sh(sh_0);
        }
        res.set_size(t, s);
        res.set_restr(flag);
        res.set_dense(sdense_0);

        return res;
    }

    // SumExpression_fast restrict(const char transa, const char transb, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) const {
    //     bool flag = true;
    //     std::vector<Matrix<CoefficientPrecision>> sr_0;
    //     Matrix<CoefficientPrecision> sdense_0;
    //     // truncation SR
    //     if (Sr.size() > 0) {
    //         Matrix<CoefficientPrecision> U = Sr[0];
    //         Matrix<CoefficientPrecision> V = Sr[1];
    //         Matrix<CoefficientPrecision> Urestr(t.get_size(), U.nb_cols());
    //         Matrix<CoefficientPrecision> Vrestr(V.nb_rows(), s.get_size());
    //         for (int k = 0; k < t.get_size(); ++k) {
    //             Urestr.set_row(k, U.get_row(k + t.get_offset() - target_offset));
    //         }
    //         for (int l = 0; l < s.get_size(); ++l) {
    //             Vrestr.set_col(l, V.get_col(l + s.get_offset() - source_offset));
    //         }
    //         std::vector<Matrix<CoefficientPrecision>> temp(2);
    //         temp[0] = Urestr;
    //         temp[1] = Vrestr;

    //         sr_0 = temp;
    //     }
    //     if (Sdense.nb_cols() > 0) {
    //         sdense_0 = copy_sub_matrix(Sdense, t.get_size(), s.get_size(), t.get_offset() - target_offset, s.get_offset() - source_offset);
    //     }
    //     std::vector<HMatrixType *> sh_0;
    //     SumExpression_fast res;
    //     // On parcours Sh si un des deux admissible on fait un ACA avec une sumexpr temporaire qui a deux matrices lw rank SR+ HK (H ou K lr)
    //     for (int k = 0; k < Sh.size() / 2; ++k) {
    //         auto &H                              = Sh[2 * k];
    //         auto &K                              = Sh[2 * k + 1];
    //         Cluster<CoefficientPrecision> source = H->get_source_cluster();
    //         if (transa == 'T') {
    //             true_source = H->get_target_cluster();
    //         }
    //         for (auto &rho_child : true_source.get_children()) {
    //             bool test_target, test_source;
    //             if (transa == 'N') {
    //                 bool test_target = H->compute_admissibility(t, *rho_child);

    //             } else {
    //                 bool test_target = H->compute_admissibility(*rho_child, t);
    //             }
    //             if (transb == 'N') {
    //                 bool test_source = H->compute_admissibility(*rho_child, s);
    //             } else {
    //                 bool test_source = H->compute_admissibility(s, *rho_child);
    //             }
    //             HMatrix<CoefficientPrecision, CoordinatePrecision> *H_child = nullptr;
    //             HMatrix<CoefficientPrecision, CoordinatePrecision> *K_child = nullptr;
    //             if (transa == 'N') {
    //                 H_child = H->get_block(t.get_size(), rho_child->get_size(), t.get_offset(), rho_child->get_offset());
    //             } else {
    //                 H_child = H->get_block(rho_child->get_size(), t.get_size(), rho_child->get_offset(), t.get_offset());
    //             }
    //             if (transb == 'N') {
    //                 K_child = K->get_block(rho_child->get_size(), s.get_size(), rho_child->get_offset(), s.get_offset());
    //             } else {
    //                 K_child = K->get_block(s.get_size(), rho_child->get_size(), s.get_offset(), rho_child->get_offset());
    //             }
    //                 if ((H_child->is_low_rank() || K_child->is_low_rank() || test_target || test_source) && (t.get_children().size() > 0) && (s.get_children().size() > 0)) {
    //                     // Compute low-rank matrix UV =  LR(HK) ;
    //                     SumExpression_fast temp(H_child, K_child);
    //                     if (sr_0.size() == 0) {
    //                         // Première affectation
    //                         LRtype lr_new(temp, *H_child->get_low_rank_generator(), t, s, -1, H_child->get_epsilon());
    //                         if (lr_new.get_U().nb_cols() > 0) {
    //                             // ACA a marché
    //                             std::vector<Matrix<CoefficientPrecision>> tempm(2);
    //                             tempm[0] = lr_new.get_U();
    //                             tempm[1] = lr_new.get_V();
    //                             sr_0     = tempm;

    //                         } else {
    //                             // on push les hmat
    //                             if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
    //                                 flag = false;
    //                             }
    //                             sh_0.push_back(H_child);
    //                             sh_0.push_back(K_child);
    //                         }
    //                     } else {
    //                         // il faut rajouter HK a SR, on appel fait un ACA sur SUMEXPR(HK, SR)
    //                         SumExpression_fast s_intermediaire(H_child, K_child);
    //                         std::vector<Matrix<CoefficientPrecision>> data_lr(2);
    //                         data_lr[0] = sr_0[0];
    //                         data_lr[1] = sr_0[1];
    //                         s_intermediaire.set_sr(data_lr, t.get_offset(), s.get_offset());
    //                         LRtype lr_bourrin(s_intermediaire, *H_child->get_low_rank_generator(), t, s, -1, H_child->get_epsilon());
    //                         if (lr_bourrin.get_U().nb_cols() > 0) {
    //                             // ACA a marché
    //                             std::vector<Matrix<CoefficientPrecision>> tempm(2);
    //                             tempm[0] = lr_bourrin.get_U();
    //                             tempm[1] = lr_bourrin.get_V();
    //                             sr_0     = tempm;

    //                         } else {
    //                             // on push les matrice hiérachique
    //                             if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
    //                                 flag = false;
    //                             }
    //                             sh_0.push_back(H_child);
    //                             sh_0.push_back(K_child);
    //                         }
    //                     }

    //                 } else {
    //                     // Pas admissble , on push les hmat
    //                     if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
    //                         flag = false;
    //                     }
    //                     sh_0.push_back(H_child);
    //                     sh_0.push_back(K_child);
    //                 }
    //             }

    //     }
    //     if (sr_0.size() > 0) {
    //         int oft = t.get_offset();
    //         int ofs = s.get_offset();
    //         res.set_sr(sr_0, oft, ofs);
    //     }
    //     if (sh_0.size() > 0) {
    //         res.set_sh(sh_0);
    //     }
    //     res.set_size(t, s);
    //     res.set_restr(flag);
    //     res.set_dense(sdense_0);

    //     return res;
    // }

    Matrix<CoefficientPrecision>
    evaluate() {
        Matrix<CoefficientPrecision> res(target_size, source_size);
        for (int k = 0; k < Sr.size() / 2; ++k) {
            CoefficientPrecision alpha = 1.0;
            CoefficientPrecision beta  = 1.0;
            int col                    = Sr[2 * k].nb_cols();
            Blas<CoefficientPrecision>::gemm("N", "N", &target_size, &source_size, &col, &alpha, Sr[2 * k].data(), &target_size, Sr[2 * k + 1].data(), &col, &beta, res.data(), &target_size);
            // res = res + (Sr[0] * Sr[1]);
            // std::cout << "sr ok" << std::endl;
        }
        for (int k = 0; k < Sh.size() / 2; ++k) {
            CoefficientPrecision alpha = 1.0;
            CoefficientPrecision beta  = 1.0;
            int col                    = Sh[2 * k]->get_source_cluster().get_size();
            Matrix<CoefficientPrecision> Hdense(target_size, col);
            Matrix<CoefficientPrecision> Kdense(col, source_size);
            if (Sh[2 * k]->is_dense() && Sh[2 * k + 1]->is_dense()) {
                Blas<CoefficientPrecision>::gemm("N", "N", &target_size, &source_size, &col, &alpha, Sh[2 * k]->get_dense_data()->data(), &target_size, Sh[2 * k + 1]->get_dense_data()->data(), &col, &beta, res.data(), &target_size);
            } else {
                auto &hh = Sh[2 * k];
                auto &kk = Sh[2 * k + 1];
                copy_to_dense(*Sh[2 * k], Hdense.data());
                copy_to_dense(*Sh[2 * k + 1], Kdense.data());
                Blas<CoefficientPrecision>::gemm("N", "N", &target_size, &source_size, &col, &alpha, Hdense.data(), &target_size, Kdense.data(), &col, &beta, res.data(), &target_size);
            }
        }
        return res;
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
class SumExpression_NT : public VirtualGenerator<CoefficientPrecision> {
  private:
    using HMatrixType = HMatrix<CoefficientPrecision, CoordinatePrecision>;
    using LRtype      = LowRankMatrix<CoefficientPrecision, CoordinatePrecision>;

    // one to rule theim all
    std::vector<Matrix<CoefficientPrecision>> Sr;
    std::vector<HMatrixType *> Sh;       //// sum (HK)
    Matrix<CoefficientPrecision> Sdense; // pour gérer les petites denses
    int target_size;
    int target_offset;
    int source_size;
    int source_offset;
    bool is_restrictible;
    char TRANSA;
    char TRANSB;

  public:
    SumExpression_NT()
        : target_size(0), target_offset(0), source_size(0), source_offset(0), TRANSA('N'), TRANSB('N') {
    }

    SumExpression_NT(const char &transa, const char &transb)
        : target_size(0), target_offset(0), source_size(0), source_offset(0), TRANSA(transa), TRANSB(transb) {
    }
    // SumExpression_fast() {
    // }
    SumExpression_NT(const char transa, HMatrixType *A, const char transb, HMatrixType *B) {
        Sh.push_back(A);
        Sh.push_back(B);
        target_size     = A->get_target_cluster().get_size();
        target_offset   = A->get_target_cluster().get_offset();
        source_size     = B->get_source_cluster().get_size();
        source_offset   = B->get_source_cluster().get_offset();
        is_restrictible = ((A->get_children().size() > 0) && (B->get_children().size()) > 0);
        TRANSA          = transa;
        TRANSB          = transb;
    }

    SumExpression_NT(const char transa, const HMatrixType *A, const char transb, const HMatrixType *B) {
        Sh.push_back(A);
        Sh.push_back(B);
        target_size     = A->get_target_cluster().get_size();
        target_offset   = A->get_target_cluster().get_offset();
        source_size     = B->get_source_cluster().get_size();
        source_offset   = B->get_source_cluster().get_offset();
        is_restrictible = ((A->get_children().size() > 0) && (B->get_children().size()) > 0);
        TRANSA          = transa;
        TRANSB          = transb;
    }

    SumExpression_NT(const char transa, const char transb, const std::vector<Matrix<CoefficientPrecision>> &sr, const std::vector<const HMatrixType *> &sh, const int &target_size0, const int &source_size0, const int &target_offset0, const int &source_offset0) {
        Sr            = sr;
        Sh            = sh;
        target_size   = target_size0;
        target_offset = target_offset0;
        source_size   = source_size0;
        source_offset = source_offset0;
        TRANSA        = transa;
        TRANSB        = transb;
    }
    SumExpression_NT<CoefficientPrecision, CoordinatePrecision> &operator=(const SumExpression_fast<CoefficientPrecision, CoordinatePrecision> &other) {
        if (this != &other) { // Vérification de l'auto-affectation
            Sr              = other.Sr;
            Sh              = other.Sh;
            target_size     = other.target_size;
            target_offset   = other.target_offset;
            source_size     = other.source_size;
            source_offset   = other.source_offset;
            TRANSA          = other.TRANSA;
            TRANSB          = other.TRANSB;
            is_restrictible = other.is_restrictible;
        }
        return *this;
    }
    ////////////////////////////
    //// Getter
    const std::vector<Matrix<CoefficientPrecision>> get_sr() const { return Sr; }
    const std::vector<HMatrixType *> get_sh() const { return Sh; }
    int get_nr() const { return target_size; }
    int get_nc() const { return source_size; }
    int get_target_offset() const { return target_offset; }
    int get_source_offset() const { return source_offset; }
    int get_target_size() const { return target_size; }
    int get_source_size() const { return source_size; }

    /////////////////////////
    //// Setter
    void set_dense(const Matrix<CoefficientPrecision> &sdense) {
        Sdense = sdense;
    }
    void set_restr(bool &flag) {
        is_restrictible = flag;
    }
    void set_sr(std::vector<Matrix<CoefficientPrecision>> &sr, const int &target_0, const int &source_0) {
        Sr            = sr;
        target_offset = target_0;
        source_offset = source_0;
        target_size   = sr[0].nb_rows();
        source_size   = sr[1].nb_cols();
    }

    void set_sh(std::vector<HMatrixType *> &sh0) { Sh = sh0; }
    void set_size(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) {
        target_offset = t.get_offset();
        source_offset = s.get_offset();
        target_size   = t.get_size();
        source_size   = s.get_size();
    }

    bool get_restr() const { return is_restrictible; }
    /////////////////////////////////////////////////////////////////////////
    /// PRODUCT SUMEXPR vect
    ////////////////////////////////////////////////////////////

    std::vector<CoefficientPrecision> prod(const char T, const std::vector<CoefficientPrecision> &x) const {
        if (T == 'N') {
            std::vector<CoefficientPrecision> y(target_size, 0.0);
            if (Sr.size() > 0) {
                for (int k = 0; k < Sr.size() / 2; ++k) {
                    auto u = Sr[2 * k];
                    auto v = Sr[2 * k + 1];
                    std::vector<CoefficientPrecision> ytemp(v.nb_rows(), 0.0);
                    v.add_vector_product('N', 1.0, x.data(), 0.0, ytemp.data());
                    u.add_vector_product('N', 1.0, ytemp.data(), 1.0, y.data());
                }
            }

            for (int k = 0; k < Sh.size() / 2; ++k) {
                auto H = Sh[2 * k];
                auto K = Sh[2 * k + 1];
                std::vector<CoefficientPrecision> y_temp(K->get_target_cluster().get_size(), 0.0);

                if (TRANSB == 'N') {
                    K->add_vector_product('N', 1.0, x.data(), 0.0, y_temp.data());
                } else {
                    y_temp.resize(K->get_source_cluster().get_size());
                    K->add_vector_product('T', 1.0, x.data(), 0.0, y_temp.data());
                }
                if (TRANSA == 'N') {
                    H->add_vector_product('N', 1.0, y_temp.data(), 1.0, y.data());
                } else {
                    H->add_vector_product('T', 1.0, y_temp.data(), 1.0, y.data());
                }
            }
            if (Sdense.nb_cols() > 0) {
                Sdense.add_vector_product('N', 1.0, x.data(), 1.0, y.data());
            }
            return (y);
        }

        else {
            std::vector<CoefficientPrecision> y(source_size, 0.0);
            if (Sr.size() > 0) {
                for (int k = 0; k < Sr.size() / 2; ++k) {

                    auto u = Sr[2 * k];
                    auto v = Sr[2 * k + 1];
                    std::vector<CoefficientPrecision> ytemp(u.nb_cols(), 0.0);

                    u.add_vector_product('T', 1.0, x.data(), 0.0, ytemp.data());
                    v.add_vector_product('T', 1.0, ytemp.data(), 1.0, y.data());
                }
            }

            // y = y +(x* urestr) * vrestrx;

            for (int k = 0; k < Sh.size() / 2; ++k) {
                auto H = Sh[2 * k];
                auto K = Sh[2 * k + 1];
                std::vector<CoefficientPrecision> y_temp(H->get_source_cluster().get_size(), 0.0);
                if (TRANSA == 'N') {
                    H->add_vector_product('T', 1.0, x.data(), 0.0, y_temp.data());
                } else {
                    y_temp.resize(H->get_target_cluster().get_size());
                    H->add_vector_product('N', 1.0, x.data(), 0.0, y_temp.data());
                }
                if (TRANSB == 'N') {
                    K->add_vector_product('T', 1.0, y_temp.data(), 1.0, y.data());
                } else {
                    K->add_vector_product('N', 1.0, y_temp.data(), 1.0, y.data());
                }
            }
            if (Sdense.nb_cols() > 0) {
                Sdense.add_vector_product('T', 1.0, x.data(), 1.0, y.data());
            }
            return (y);
        }
    }

    ////////////
    /// get _row & get_col by performing Sum_expr vector multiplication
    std::vector<CoefficientPrecision>
    get_col(const int &l) const {
        std::vector<double> e_l(source_size);
        e_l[l]   = 1.0;
        auto col = this->prod('N', e_l);
        return col;
    }
    std::vector<CoefficientPrecision> get_row(const int &k) const {
        std::vector<double> e_k(target_size);
        e_k[k]   = 1.0;
        auto row = this->prod('T', e_k);
        return row;
    }
    //////////////////////////////////////////
    //// COp submatrix
    //////////////////////////////////////
    // double norme_approx() {
    //     double nr = 0.0;
    // }
    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        // on regarde si il faut pas juste une ligne ou une collonne
        Matrix<CoefficientPrecision> mat(M, N);
        mat.assign(M, N, ptr, false);
        int ref_row = row_offset - target_offset;
        int ref_col = col_offset - source_offset;
        if (M == 1) {
            auto row = this->get_row(ref_row);
            std::copy(row.begin() + ref_col, row.begin() + ref_col + N, mat.data());
        } else if (N == 1) {
            auto col = this->get_col(ref_col);
            std::copy(col.begin() + ref_row, col.begin() + ref_row + M, mat.data());
        }

        else {
            // on veut toute la matrice
            for (int k = 0; k < Sh.size() / 2; ++k) {
                auto &H             = Sh[2 * k];
                auto &K             = Sh[2 * k + 1];
                int row_H           = H->get_target_cluster().get_size();
                int col_K           = K->get_source_cluster().get_size();
                int intermediaire_H = H->get_source_cluster().get_size();
                int intermediaire_K = K->get_target_cluster().get_size();
                if (TRANSA == 'T') {
                    intermediaire_H = H->get_target_cluster().get_size();
                    row_H           = H->get_source_cluster().get_size();
                }
                if (TRANSB == 'T') {
                    intermediaire_K = K->get_source_cluster().get_size();
                    col_K           = K->get_target_cluster().get_size();
                }

                if (row_H != target_size || col_K != source_size) {
                    std::cout << "Cas pas normal : H et K pas allignés" << std::endl;
                }
                Matrix<CoefficientPrecision> hdense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                Matrix<CoefficientPrecision> kdense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                copy_to_dense(*H, hdense.data());
                copy_to_dense(*K, kdense.data());

                int m                      = target_size;
                int n                      = source_size;
                int kk                     = intermediaire_H;
                CoefficientPrecision alpha = 1.0;
                CoefficientPrecision beta  = 1.0;
                int lda                    = m;
                int ldb                    = kk;
                int ldc                    = m;
                Blas<CoefficientPrecision>::gemm(&TRANSA, &TRANSB, &m, &n, &kk, &alpha, hdense.data(), &lda, kdense.data(), &ldb, &beta, mat.data(), &ldc);
                // mat = mat + hrestr * krestr;
            }
            if (Sr.size() > 0) {
                for (int k = 0; k < Sr.size() / 2; ++k) {
                    auto U = Sr[2 * k];
                    auto V = Sr[2 * k + 1];
                    Matrix<CoefficientPrecision> U_restr(target_size, U.nb_cols());
                    Matrix<CoefficientPrecision> V_restr(V.nb_rows(), source_size);
                    for (int i = 0; i < target_size; ++i) {
                        U_restr.set_row(i, U.get_row(i + row_offset - target_offset));
                    }
                    for (int j = 0; j < source_size; ++j) {
                        V_restr.set_col(j, V.get_col(j + col_offset - source_offset));
                    }
                    U_restr.add_matrix_product('N', 1.0, V_restr.data(), 1.0, mat.data(), N);
                    // mat = mat + U_restr * V_restr;
                }
            }
            if (Sdense.nb_cols() > 0) {
                mat.plus_egal(Sdense);
                // mat = mat + copy_sub_matrix(Sdense, M, N, ref_row, ref_col);
            }
        }
        std::copy(mat.data(), mat.data() + M * N, ptr);
    }

    // // do Sr = Sr +AB with QR svd
    // // U1V1^T+U2V2^T = (U1, U2)(V1,V2)^T = (Q1R1)(Q2R2)T =

    SumExpression_NT restrict(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) const {
        bool flag = true;
        std::vector<Matrix<CoefficientPrecision>> sr_0;
        Matrix<CoefficientPrecision> sdense_0;
        // truncation SR
        if (Sr.size() > 0) {
            Matrix<CoefficientPrecision> U = Sr[0];
            Matrix<CoefficientPrecision> V = Sr[1];
            Matrix<CoefficientPrecision> Urestr(t.get_size(), U.nb_cols());
            Matrix<CoefficientPrecision> Vrestr(V.nb_rows(), s.get_size());
            for (int k = 0; k < t.get_size(); ++k) {
                Urestr.set_row(k, U.get_row(k + t.get_offset() - target_offset));
            }
            for (int l = 0; l < s.get_size(); ++l) {
                Vrestr.set_col(l, V.get_col(l + s.get_offset() - source_offset));
            }
            std::vector<Matrix<CoefficientPrecision>> temp(2);
            temp[0] = Urestr;
            temp[1] = Vrestr;

            sr_0 = temp;
        }
        if (Sdense.nb_cols() > 0) {
            sdense_0 = copy_sub_matrix(Sdense, t.get_size(), s.get_size(), t.get_offset() - target_offset, s.get_offset() - source_offset);
        }
        std::vector<HMatrixType *> sh_0;
        SumExpression_NT res(TRANSA, TRANSB);
        // On parcours Sh si un des deux admissible on fait un ACA avec une sumexpr temporaire qui a deux matrices lw rank SR+ HK (H ou K lr)
        for (int k = 0; k < Sh.size() / 2; ++k) {
            auto &H                    = Sh[2 * k];
            auto &K                    = Sh[2 * k + 1];
            const auto &source_cluster = H->get_source_cluster();
            const auto &target_cluster = H->get_target_cluster();

            const auto &true_source = (TRANSA == 'T') ? target_cluster : source_cluster;
            for (auto &rho_child : true_source.get_children()) {
                bool test_target, test_source;
                if (TRANSA == 'N') {
                    test_target = H->compute_admissibility(t, *rho_child);

                } else {
                    test_target = H->compute_admissibility(*rho_child, t);
                }
                if (TRANSB == 'N') {
                    bool test_source = H->compute_admissibility(*rho_child, s);
                } else {
                    bool test_source = H->compute_admissibility(s, *rho_child);
                }
                HMatrix<CoefficientPrecision, CoordinatePrecision> *H_child = nullptr;
                HMatrix<CoefficientPrecision, CoordinatePrecision> *K_child = nullptr;
                if (TRANSA == 'N') {
                    H_child = H->get_block(t.get_size(), rho_child->get_size(), t.get_offset(), rho_child->get_offset());
                } else {
                    H_child = H->get_block(rho_child->get_size(), t.get_size(), rho_child->get_offset(), t.get_offset());
                }
                if (TRANSB == 'N') {
                    K_child = K->get_block(rho_child->get_size(), s.get_size(), rho_child->get_offset(), s.get_offset());
                } else {
                    K_child = K->get_block(s.get_size(), rho_child->get_size(), s.get_offset(), rho_child->get_offset());
                }
                if ((H_child->is_low_rank() || K_child->is_low_rank() || test_target || test_source) && (t.get_children().size() > 0) && (s.get_children().size() > 0)) {
                    // Compute low-rank matrix UV =  LR(HK) ;
                    SumExpression_NT temp(TRANSA, H_child, TRANSB, K_child);
                    if (sr_0.size() == 0) {
                        // Première affectation
                        LRtype lr_new(temp, *H_child->get_low_rank_generator(), t, s, -1, H_child->get_epsilon());
                        if (lr_new.get_U().nb_cols() > 0) {
                            // ACA a marché
                            std::vector<Matrix<CoefficientPrecision>> tempm(2);
                            tempm[0] = lr_new.get_U();
                            tempm[1] = lr_new.get_V();
                            sr_0     = tempm;

                        } else {
                            // on push les hmat
                            if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
                                flag = false;
                            }
                            sh_0.push_back(H_child);
                            sh_0.push_back(K_child);
                        }
                    } else {
                        // il faut rajouter HK a SR, on appel fait un ACA sur SUMEXPR(HK, SR)
                        SumExpression_NT s_intermediaire(TRANSA, H_child, TRANSB, K_child);
                        std::vector<Matrix<CoefficientPrecision>> data_lr(2);
                        data_lr[0] = sr_0[0];
                        data_lr[1] = sr_0[1];
                        s_intermediaire.set_sr(data_lr, t.get_offset(), s.get_offset());
                        LRtype lr_bourrin(s_intermediaire, *H_child->get_low_rank_generator(), t, s, -1, H_child->get_epsilon());
                        if (lr_bourrin.get_U().nb_cols() > 0) {
                            // ACA a marché
                            std::vector<Matrix<CoefficientPrecision>> tempm(2);
                            tempm[0] = lr_bourrin.get_U();
                            tempm[1] = lr_bourrin.get_V();
                            sr_0     = tempm;

                        } else {
                            // on push les matrice hiérachique
                            if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
                                flag = false;
                            }
                            sh_0.push_back(H_child);
                            sh_0.push_back(K_child);
                        }
                    }

                } else {
                    // Pas admissble , on push les hmat
                    if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
                        flag = false;
                    }
                    sh_0.push_back(H_child);
                    sh_0.push_back(K_child);
                }
            }
        }
        if (sr_0.size() > 0) {
            int oft = t.get_offset();
            int ofs = s.get_offset();
            res.set_sr(sr_0, oft, ofs);
        }
        if (sh_0.size() > 0) {
            res.set_sh(sh_0);
        }
        res.set_size(t, s);
        res.set_restr(flag);
        res.set_dense(sdense_0);

        return res;
    }

    // SumExpression_fast restrict_ACA_triangulaire(char transa, char transb, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) const {
    //     bool flag = true;
    //     std::vector<Matrix<CoefficientPrecision>> sr_0;
    //     Matrix<CoefficientPrecision> sdense_0;
    //     // truncation SR
    //     if (Sr.size() > 0) {
    //         Matrix<CoefficientPrecision> U = Sr[0];
    //         Matrix<CoefficientPrecision> V = Sr[1];
    //         Matrix<CoefficientPrecision> Urestr(t.get_size(), U.nb_cols());
    //         Matrix<CoefficientPrecision> Vrestr(V.nb_rows(), s.get_size());
    //         for (int k = 0; k < t.get_size(); ++k) {
    //             Urestr.set_row(k, U.get_row(k + t.get_offset() - target_offset));
    //         }
    //         for (int l = 0; l < s.get_size(); ++l) {
    //             Vrestr.set_col(l, V.get_col(l + s.get_offset() - source_offset));
    //         }
    //         std::vector<Matrix<CoefficientPrecision>> temp(2);
    //         temp[0] = Urestr;
    //         temp[1] = Vrestr;

    //         sr_0 = temp;
    //     }
    //     if (Sdense.nb_cols() > 0) {
    //         sdense_0 = copy_sub_matrix(Sdense, t.get_size(), s.get_size(), t.get_offset() - target_offset, s.get_offset() - source_offset);
    //     }
    //     std::vector<HMatrixType *> sh_0;
    //     SumExpression_fast res;
    //     // On parcours Sh si un des deux admissible on fait un ACA avec une sumexpr temporaire qui a deux matrices lw rank SR+ HK (H ou K lr)
    //     for (int k = 0; k < Sh.size() / 2; ++k) {
    //         auto &H = Sh[2 * k];
    //         auto &K = Sh[2 * k + 1];
    //         for (auto &rho_child : H->get_source_cluster().get_children()) {
    //             if (transa == 'L') {
    //                 if (rho_child->get_offset() > t.get_offset()) {
    //                     continue;
    //                 }
    //             } else if (transa == 'U') {
    //                 if (rho_child->get_offset() < t.get_offset()) {
    //                     continue;
    //                 }
    //             }
    //             if (transb == 'L') {
    //                 if (rho_child->get_offset() < s.get_offset()) {
    //                     continue;
    //                 }
    //             } else if (transb == 'U') {
    //                 if (rho_child->get_offset() > s.get_offset()) {
    //                     continue;
    //                 }
    //             }
    //             bool test_target = H->compute_admissibility(t, *rho_child);
    //             bool test_source = H->compute_admissibility(*rho_child, s);

    //             auto H_child = H->get_block(t.get_size(), rho_child->get_size(), t.get_offset(), rho_child->get_offset());
    //             auto K_child = K->get_block(rho_child->get_size(), s.get_size(), rho_child->get_offset(), s.get_offset());
    //             // si on arrive pas a choper les blocs : ca doit pas arriver mais bon
    //             if (H_child->get_target_cluster().get_size() != t.get_size() || H_child->get_source_cluster().get_size() != rho_child->get_size() || K_child->get_target_cluster().get_size() != rho_child->get_size() || K_child->get_source_cluster().get_size() != s.get_size()) {
    //                 // std::cout << "?! " << std::endl;
    //                 // std::cout << H_child->get_target_cluster().get_size() << '=' << t.get_size() << ',' << H_child->get_source_cluster().get_size() << '=' << rho_child->get_size() << ',' << K_child->get_target_cluster().get_size() << '=' << rho_child->get_size() << ',' << K_child->get_source_cluster().get_size() << '=' << s.get_size() << std::endl;
    //                 Matrix<CoefficientPrecision> hdense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
    //                 Matrix<CoefficientPrecision> kdense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
    //                 copy_to_dense(*H, hdense.data());
    //                 copy_to_dense(*K, kdense.data());
    //                 std::cout << "            avant d'extraire : " << normFrob(hdense) << ',' << normFrob(kdense) << std::endl;
    //                 auto hrestr                = copy_sub_matrix(hdense, t.get_size(), rho_child->get_size(), t.get_offset() - H->get_target_cluster().get_offset(), rho_child->get_offset() - H->get_source_cluster().get_offset());
    //                 auto krestr                = copy_sub_matrix(kdense, rho_child->get_size(), s.get_size(), rho_child->get_offset() - K->get_target_cluster().get_offset(), s.get_offset() - K->get_source_cluster().get_offset());
    //                 const char trans           = 'N';
    //                 int m                      = t.get_size();
    //                 int n                      = s.get_size();
    //                 int kk                     = rho_child->get_size();
    //                 CoefficientPrecision alpha = 1.0;
    //                 CoefficientPrecision beta  = 1.0;
    //                 int lda                    = m;
    //                 int ldb                    = kk;
    //                 int ldc                    = m;
    //                 if (sdense_0.nb_cols() == 0) {
    //                     sdense_0.resize(t.get_size(), s.get_size());
    //                 }
    //                 Blas<CoefficientPrecision>::gemm(&trans, &trans, &m, &n, &kk, &alpha, hrestr.data(), &lda, krestr.data(), &ldb, &beta, sdense_0.data(), &ldc);
    //                 std::cout << "sdense de norme : " << normFrob(sdense_0) << ',' << sdense_0.nb_rows() << ',' << sdense_0.nb_cols() << std::endl;
    //                 flag = false;

    //             } else {
    //                 // sinon on regarde si ils sont admissibles
    //                 if ((H_child->is_low_rank() || K_child->is_low_rank() || test_target || test_source) && (t.get_children().size() > 0) && (s.get_children().size() > 0)) {
    //                     // Compute low-rank matrix UV =  LR(HK) ;
    //                     SumExpression_fast temp(H_child, K_child);
    //                     if (sr_0.size() == 0) {
    //                         // Première affectation
    //                         LRtype lr_new(temp, *H_child->get_low_rank_generator(), t, s, -1, H_child->get_epsilon());
    //                         if (lr_new.get_U().nb_cols() > 0) {
    //                             // ACA a marché
    //                             std::vector<Matrix<CoefficientPrecision>> tempm(2);
    //                             tempm[0] = lr_new.get_U();
    //                             tempm[1] = lr_new.get_V();
    //                             sr_0     = tempm;

    //                         } else {
    //                             // on push les hmat
    //                             if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
    //                                 flag = false;
    //                             }
    //                             sh_0.push_back(H_child);
    //                             sh_0.push_back(K_child);
    //                         }
    //                     } else {
    //                         // il faut rajouter HK a SR, on appel fait un ACA sur SUMEXPR(HK, SR)
    //                         SumExpression_fast s_intermediaire(H_child, K_child);
    //                         std::vector<Matrix<CoefficientPrecision>> data_lr(2);
    //                         data_lr[0] = sr_0[0];
    //                         data_lr[1] = sr_0[1];
    //                         s_intermediaire.set_sr(data_lr, t.get_offset(), s.get_offset());
    //                         LRtype lr_bourrin(s_intermediaire, *H_child->get_low_rank_generator(), t, s, -1, H_child->get_epsilon());
    //                         if (lr_bourrin.get_U().nb_cols() > 0) {
    //                             // ACA a marché
    //                             std::vector<Matrix<CoefficientPrecision>> tempm(2);
    //                             tempm[0] = lr_bourrin.get_U();
    //                             tempm[1] = lr_bourrin.get_V();
    //                             sr_0     = tempm;

    //                         } else {
    //                             // on push les matrice hiérachique
    //                             if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
    //                                 flag = false;
    //                             }
    //                             sh_0.push_back(H_child);
    //                             sh_0.push_back(K_child);
    //                         }
    //                     }

    //                 } else {
    //                     // Pas admissble , on push les hmat
    //                     if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
    //                         flag = false;
    //                     }
    //                     sh_0.push_back(H_child);
    //                     sh_0.push_back(K_child);
    //                 }
    //             }
    //         }
    //     }
    //     if (sr_0.size() > 0) {
    //         int oft = t.get_offset();
    //         int ofs = s.get_offset();
    //         res.set_sr(sr_0, oft, ofs);
    //     }
    //     if (sh_0.size() > 0) {
    //         res.set_sh(sh_0);
    //     }
    //     res.set_size(t, s);
    //     res.set_restr(flag);
    //     res.set_dense(sdense_0);

    //     return res;
    // }

    // // SumExpression_fast restrict(const char transa, const char transb, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) const {
    // //     bool flag = true;
    // //     std::vector<Matrix<CoefficientPrecision>> sr_0;
    // //     Matrix<CoefficientPrecision> sdense_0;
    // //     // truncation SR
    // //     if (Sr.size() > 0) {
    // //         Matrix<CoefficientPrecision> U = Sr[0];
    // //         Matrix<CoefficientPrecision> V = Sr[1];
    // //         Matrix<CoefficientPrecision> Urestr(t.get_size(), U.nb_cols());
    // //         Matrix<CoefficientPrecision> Vrestr(V.nb_rows(), s.get_size());
    // //         for (int k = 0; k < t.get_size(); ++k) {
    // //             Urestr.set_row(k, U.get_row(k + t.get_offset() - target_offset));
    // //         }
    // //         for (int l = 0; l < s.get_size(); ++l) {
    // //             Vrestr.set_col(l, V.get_col(l + s.get_offset() - source_offset));
    // //         }
    // //         std::vector<Matrix<CoefficientPrecision>> temp(2);
    // //         temp[0] = Urestr;
    // //         temp[1] = Vrestr;

    // //         sr_0 = temp;
    // //     }
    // //     if (Sdense.nb_cols() > 0) {
    // //         sdense_0 = copy_sub_matrix(Sdense, t.get_size(), s.get_size(), t.get_offset() - target_offset, s.get_offset() - source_offset);
    // //     }
    // //     std::vector<HMatrixType *> sh_0;
    // //     SumExpression_fast res;
    // //     // On parcours Sh si un des deux admissible on fait un ACA avec une sumexpr temporaire qui a deux matrices lw rank SR+ HK (H ou K lr)
    // //     for (int k = 0; k < Sh.size() / 2; ++k) {
    // //         auto &H                              = Sh[2 * k];
    // //         auto &K                              = Sh[2 * k + 1];
    // //         Cluster<CoefficientPrecision> source = H->get_source_cluster();
    // //         if (transa == 'T') {
    // //             true_source = H->get_target_cluster();
    // //         }
    // //         for (auto &rho_child : true_source.get_children()) {
    // //             bool test_target, test_source;
    // //             if (transa == 'N') {
    // //                 bool test_target = H->compute_admissibility(t, *rho_child);

    // //             } else {
    // //                 bool test_target = H->compute_admissibility(*rho_child, t);
    // //             }
    // //             if (transb == 'N') {
    // //                 bool test_source = H->compute_admissibility(*rho_child, s);
    // //             } else {
    // //                 bool test_source = H->compute_admissibility(s, *rho_child);
    // //             }
    // //             HMatrix<CoefficientPrecision, CoordinatePrecision> *H_child = nullptr;
    // //             HMatrix<CoefficientPrecision, CoordinatePrecision> *K_child = nullptr;
    // //             if (transa == 'N') {
    // //                 H_child = H->get_block(t.get_size(), rho_child->get_size(), t.get_offset(), rho_child->get_offset());
    // //             } else {
    // //                 H_child = H->get_block(rho_child->get_size(), t.get_size(), rho_child->get_offset(), t.get_offset());
    // //             }
    // //             if (transb == 'N') {
    // //                 K_child = K->get_block(rho_child->get_size(), s.get_size(), rho_child->get_offset(), s.get_offset());
    // //             } else {
    // //                 K_child = K->get_block(s.get_size(), rho_child->get_size(), s.get_offset(), rho_child->get_offset());
    // //             }
    // //                 if ((H_child->is_low_rank() || K_child->is_low_rank() || test_target || test_source) && (t.get_children().size() > 0) && (s.get_children().size() > 0)) {
    // //                     // Compute low-rank matrix UV =  LR(HK) ;
    // //                     SumExpression_fast temp(H_child, K_child);
    // //                     if (sr_0.size() == 0) {
    // //                         // Première affectation
    // //                         LRtype lr_new(temp, *H_child->get_low_rank_generator(), t, s, -1, H_child->get_epsilon());
    // //                         if (lr_new.get_U().nb_cols() > 0) {
    // //                             // ACA a marché
    // //                             std::vector<Matrix<CoefficientPrecision>> tempm(2);
    // //                             tempm[0] = lr_new.get_U();
    // //                             tempm[1] = lr_new.get_V();
    // //                             sr_0     = tempm;

    // //                         } else {
    // //                             // on push les hmat
    // //                             if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
    // //                                 flag = false;
    // //                             }
    // //                             sh_0.push_back(H_child);
    // //                             sh_0.push_back(K_child);
    // //                         }
    // //                     } else {
    // //                         // il faut rajouter HK a SR, on appel fait un ACA sur SUMEXPR(HK, SR)
    // //                         SumExpression_fast s_intermediaire(H_child, K_child);
    // //                         std::vector<Matrix<CoefficientPrecision>> data_lr(2);
    // //                         data_lr[0] = sr_0[0];
    // //                         data_lr[1] = sr_0[1];
    // //                         s_intermediaire.set_sr(data_lr, t.get_offset(), s.get_offset());
    // //                         LRtype lr_bourrin(s_intermediaire, *H_child->get_low_rank_generator(), t, s, -1, H_child->get_epsilon());
    // //                         if (lr_bourrin.get_U().nb_cols() > 0) {
    // //                             // ACA a marché
    // //                             std::vector<Matrix<CoefficientPrecision>> tempm(2);
    // //                             tempm[0] = lr_bourrin.get_U();
    // //                             tempm[1] = lr_bourrin.get_V();
    // //                             sr_0     = tempm;

    // //                         } else {
    // //                             // on push les matrice hiérachique
    // //                             if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
    // //                                 flag = false;
    // //                             }
    // //                             sh_0.push_back(H_child);
    // //                             sh_0.push_back(K_child);
    // //                         }
    // //                     }

    // //                 } else {
    // //                     // Pas admissble , on push les hmat
    // //                     if (H_child->is_low_rank() || K_child->is_low_rank() || t.get_children().size() == 0 || s.get_children().size() == 0) {
    // //                         flag = false;
    // //                     }
    // //                     sh_0.push_back(H_child);
    // //                     sh_0.push_back(K_child);
    // //                 }
    // //             }

    // //     }
    // //     if (sr_0.size() > 0) {
    // //         int oft = t.get_offset();
    // //         int ofs = s.get_offset();
    // //         res.set_sr(sr_0, oft, ofs);
    // //     }
    // //     if (sh_0.size() > 0) {
    // //         res.set_sh(sh_0);
    // //     }
    // //     res.set_size(t, s);
    // //     res.set_restr(flag);
    // //     res.set_dense(sdense_0);

    // //     return res;
    // // }

    // Matrix<CoefficientPrecision>
    // evaluate() {
    //     Matrix<CoefficientPrecision> res(target_size, source_size);
    //     for (int k = 0; k < Sr.size() / 2; ++k) {
    //         CoefficientPrecision alpha = 1.0;
    //         CoefficientPrecision beta  = 1.0;
    //         int col                    = Sr[2 * k].nb_cols();
    //         Blas<CoefficientPrecision>::gemm("N", "N", &target_size, &source_size, &col, &alpha, Sr[2 * k].data(), &target_size, Sr[2 * k + 1].data(), &col, &beta, res.data(), &target_size);
    //         // res = res + (Sr[0] * Sr[1]);
    //         // std::cout << "sr ok" << std::endl;
    //     }
    //     for (int k = 0; k < Sh.size() / 2; ++k) {
    //         CoefficientPrecision alpha = 1.0;
    //         CoefficientPrecision beta  = 1.0;
    //         int col                    = Sh[2 * k]->get_source_cluster().get_size();
    //         Matrix<CoefficientPrecision> Hdense(target_size, col);
    //         Matrix<CoefficientPrecision> Kdense(col, source_size);
    //         if (Sh[2 * k]->is_dense() && Sh[2 * k + 1]->is_dense()) {
    //             Blas<CoefficientPrecision>::gemm("N", "N", &target_size, &source_size, &col, &alpha, Sh[2 * k]->get_dense_data()->data(), &target_size, Sh[2 * k + 1]->get_dense_data()->data(), &col, &beta, res.data(), &target_size);
    //         } else {
    //             auto &hh = Sh[2 * k];
    //             auto &kk = Sh[2 * k + 1];
    //             copy_to_dense(*Sh[2 * k], Hdense.data());
    //             copy_to_dense(*Sh[2 * k + 1], Kdense.data());
    //             Blas<CoefficientPrecision>::gemm("N", "N", &target_size, &source_size, &col, &alpha, Hdense.data(), &target_size, Kdense.data(), &col, &beta, res.data(), &target_size);
    //         }
    //     }
    //     return res;
    // }
};

} // namespace htool
#endif
