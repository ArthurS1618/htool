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
    const CoefficientPrecision get_coeff(const int &i, const int &j) const {
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
    /// Arthur TO DO : distinction cas parce que c'est violent quand même le copy_to_dense
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
                std::cout << "énorme bloc densifié : " << M << ',' << N << " dans un bloc de taille :  " << target_size << ',' << source_size << "|" << target_offset << ',' << source_offset << std::endl;

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
                // if (M == target_size && N == source_size) {
                //     auto x = generate_random_vector(source_size);
                //     std::cout << "erreur vecteur ? " << norm2(this->new_prod('N', x) - mat * x) / norm2(mat * x) << std::endl;
                //     std::cout << "erreur vecteur T ? " << norm2(this->new_prod('T', x) - mat.transp(mat) * x) / norm2(mat.transp(mat) * x) << std::endl;
                // }

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
} // namespace htool

#endif
