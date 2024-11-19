#include <cmath>
#include <fstream>
#include <htool/basic_types/vector.hpp>
#include <htool/clustering/cluster_output.hpp>
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/linalg/interface.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>
#include <htool/wrappers/wrapper_lapack.hpp>
#include <iostream>
#include <mpi.h>
#include <regex>
#include <string>

using namespace std;
using namespace htool;

std::vector<std::vector<double>> get_csr(const std::string &matfile) {
    std::ifstream file(matfile);
    std::vector<double> data_csr;
    std::vector<double> col_csr;
    std::vector<double> row_csr;
    std::vector<std::vector<double>> res;

    if (!file.is_open()) {
        std::cerr << "Error: Can't open the file" << std::endl;
        return res;
    }

    std::string line;
    std::string word;
    char delimiter = ',';

    // Read first line
    if (getline(file, line)) {
        std::stringstream ss(line);
        while (getline(ss, word, delimiter)) {
            double d1 = std::stod(word);
            data_csr.push_back(d1);
        }
    }

    // Read second line
    if (getline(file, line)) {
        std::stringstream ss(line);
        while (getline(ss, word, delimiter)) {
            double d1 = std::stod(word);
            col_csr.push_back(d1);
        }
    }

    // Read third line
    if (getline(file, line)) {
        std::stringstream ss(line);
        while (getline(ss, word, delimiter)) {
            double d1 = std::stod(word);
            row_csr.push_back(d1);
        }
    }

    file.close();
    res.push_back(data_csr);
    res.push_back(row_csr);
    res.push_back(col_csr);
    return res;
}

template <class CoefficientPrecision, class CoordinatePrecision>
class CSR_generator : public VirtualGenerator<CoefficientPrecision> {
  private:
    std::vector<double> data;
    std::vector<double> row_csr;
    std::vector<double> col_csr;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;

  public:
    CSR_generator(const Cluster<CoordinatePrecision> &target0,
                  const Cluster<CoordinatePrecision> &source0,
                  const std::vector<double> &row_csr0,
                  const std::vector<double> &col_csr0,
                  const std::vector<double> &val_csr0) : target(target0), source(source0) {

        int M = target0.get_size(); // nombre de lignes après permutation
        int N = source0.get_size(); // nombre de colonnes après permutation

        // Initialisation des nouvelles structures CSR
        row_csr.resize(M + 1, 0);
        int rep = 0;
        // Boucle sur les lignes permutées
        for (int i = 0; i < M; i++) {
            int row_perm = target0.get_permutation()[i];
            int r1       = row_csr0[row_perm];     // début de la ligne dans row_csr
            int r2       = row_csr0[row_perm + 1]; // fin de la ligne dans row_csr
            // Boucle sur les colonnes permutées pour la ligne courante
            int flag = 0;
            for (int j = 0; j < N; j++) {
                int col_perm = source0.get_permutation()[j];
                // Extraire les colonnes et les valeurs de la ligne permutée
                std::vector<int> row_iperm(r2 - r1);
                std::copy(col_csr0.begin() + r1, col_csr0.begin() + r2, row_iperm.begin());
                // Rechercher l'indice de la colonne permutée
                auto it = std::find(row_iperm.begin(), row_iperm.end(), col_perm);
                if (it != row_iperm.end()) {
                    int idx = std::distance(row_iperm.begin(), it);
                    col_csr.push_back(j);               // colonne après permutation
                    data.push_back(val_csr0[r1 + idx]); // valeur associée
                    flag += 1;
                }
            }
            // Mettre à jour row_csr_out pour la nouvelle ligne
            row_csr[i + 1] = row_csr[i] + flag;
        }
    }

    void mat_vect(const std::vector<CoefficientPrecision> &in, std::vector<CoefficientPrecision> &out) {
        int n = row_csr.size() - 1;
        for (int i = 0; i < n; ++i) {
            for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
                out[i] += data[j] * in[col_csr[j]];
            }
        }
    }
    // void mat_vect_bloc(const int &size_t, const int &size_s, const int &oft, const int &ofs, const std::vector<CoefficientPrecision> &in, std::vector<CoefficientPrecision> &out) {
    //     for (int i = oft; i < size_t + oft; ++i) {
    //         for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
    //             int col = col_idx[j] - ofs;     // Ajuster l'indice de colonne avec ofs
    //             if (col >= 0 && col < size_s) { // Vérifier si la colonne est dans le sous-bloc
    //                 out[i] += data[j] * in[col];
    //             }
    //         }
    //     }
    // }

    void mat_vect_bloc(const int &size_t, const int &size_s, const int &oft, const int &ofs, const std::vector<CoefficientPrecision> &in, std::vector<CoefficientPrecision> &out, char trans) const {
        if (trans == 'N') {
            // Multiplication standard: CSR * vecteur
            for (int i = oft; i < size_t + oft; ++i) {
                for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
                    int col = col_csr[j] - ofs;     // Ajuster l'indice de colonne avec ofs
                    if (col >= 0 && col < size_s) { // Vérifier si la colonne est dans le sous-bloc
                        out[i] += data[j] * in[col];
                    }
                }
            }
        } else if (trans == 'T') {
            // Multiplication transposée: CSR^T * vecteur
            // Note: Pour la transposée, les rôles de `i` et `col` sont inversés.
            for (int i = oft; i < size_t + oft; ++i) {
                for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
                    int col = col_csr[j] - ofs;            // Ajuster l'indice de colonne avec ofs
                    if (col >= 0 && col < size_s) {        // Vérifier si la colonne est dans le sous-bloc
                        out[col] += data[j] * in[i - oft]; // Transposer le calcul : out[col] = A^T * in
                    }
                }
            }
        }
    }

    std::vector<Matrix<CoefficientPrecision>> rd_ACA(const int &size_t, const int &size_s, const int &oft, const int &ofs, const double &tolerance, bool &flagout) const {
        int rk = 1;
        Matrix<CoefficientPrecision> L(size_t, 1);
        Matrix<CoefficientPrecision> Ut(size_s, 1);
        Matrix<CoefficientPrecision> temp(size_t, size_t);
        std::vector<Matrix<CoefficientPrecision>> res;
        for (int k = 0; k < size_t; ++k) {
            temp(k, k) = 1.0;
        }
        double norme_LU = 0.0;
        // auto bloc       = this->copy_submatrix(size_t, size_s, oft, ofs);
        auto w = gaussian_vector(size_s, 0.0, 1.0);
        while ((rk < size_t / 2) && (rk < size_s / 2)) {
            std::vector<CoefficientPrecision> Aw(size_t, 0.0);
            // bloc->add_vector_product('N', 1.0, w.data(), 1.0, Aw.data());
            this->mat_vect_bloc(size_t, size_s, oft, ofs, w, Aw, 'N');
            std::vector<CoefficientPrecision> lr(size_t, 0.0);
            temp.add_vector_product('N', 1.0, Aw.data(), 1.0, lr.data());
            auto norm_lr = norm2(lr);
            if (norm_lr < 1e-20) {
                break;
            } else {
                lr = mult(1.0 / norm_lr, lr);
            }
            std::vector<CoefficientPrecision> ur(size_s, 0.0);
            // bloc->add_vector_product('T', 1.0, lr.data(), 1.0, ur.data());
            this->mat_vect_bloc(size_t, size_s, oft, ofs, lr, ur, 'T');
            auto norm_ur = norm2(ur);
            // std::cout << "norm2 ur : " << norm_ur << std::endl;
            Matrix<CoefficientPrecision> lr_ur(size_t, size_t);
            double alpha = 1.0;
            double beta  = -1.0;
            int kk       = 1;
            Blas<CoefficientPrecision>::gemm("N", "T", &size_t, &size_t, &kk, &alpha, lr.data(), &size_t, ur.data(), &size_t, &alpha, lr_ur.data(), &size_t);
            std::vector<CoefficientPrecision> Llr(L.nb_cols(), 0.0);
            L.add_vector_product('T', 1.0, lr.data(), 1.0, Llr.data());
            std::vector<CoefficientPrecision> ULlr(size_s, 0.0);
            Ut.add_vector_product('N', 1.0, Llr.data(), 1.0, ULlr.data());
            double trace = 0.0;
            for (int k = 0; k < std::min(size_t, size_s); ++k) {
                trace += (ULlr[k] * ur[k]);
            }
            auto nr_lrur = normFrob(lr_ur);
            auto nr      = norme_LU + std::pow(nr_lrur, 2.0) + 2 * trace;
            if (rk > 1 && (std::pow(norm_lr * norm_ur * (size_s - rk), 2.0) <= std::pow(tolerance, 2.0) * nr)) {
                break;
            }
            if (rk == 1) {
                std::copy(lr.data(), lr.data() + size_t, L.data());
                std::copy(ur.data(), ur.data() + size_s, Ut.data());
                // std::cout << "!!!!!! aprés affectation " << normFrob(L) << ',' << normFrob(Ut) << "! " << norm_lr << ',' << norm_ur << std::endl;
            } else {
                Matrix<CoefficientPrecision> new_L(size_t, rk);
                std::copy(L.data(), L.data() + L.nb_rows() * L.nb_cols(), new_L.data());
                std::copy(lr.data(), lr.data() + size_t, new_L.data() + (rk - 1) * size_t);
                Matrix<CoefficientPrecision> new_U(size_s, rk);
                std::copy(Ut.data(), Ut.data() + Ut.nb_rows() * Ut.nb_cols(), new_U.data());
                std::copy(ur.data(), ur.data() + size_s, new_U.data() + (rk - 1) * size_s);
                L  = new_L;
                Ut = new_U;
            }
            w = gaussian_vector(size_s, 0.0, 1.0);
            Blas<CoefficientPrecision>::gemm("N", "T", &size_t, &size_t, &kk, &beta, lr.data(), &size_t, lr.data(), &size_t, &alpha, temp.data(), &size_t);
            norme_LU += std::pow(nr_lrur, 2.0);
            rk += 1;
        }
        if (rk >= std::min(size_t / 2, size_s / 2)) {
            flagout = false;
        }
        if (flagout) {
            if (rk == 1) {
                Matrix<CoefficientPrecision> U(1, size_s);
                res.push_back(L);
                res.push_back(U);
            } else {
                Matrix<CoefficientPrecision> U(rk - 1, size_s);
                transpose(Ut, U);
                // LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_mat(L, U);
                // bloc->delete_children();
                // bloc->set_low_rank_data(lr_mat);
                res.push_back(L);
                res.push_back(U);
            }
        } else {
            // Matrix<CoefficientPrecision> dense(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
            // copy_to_dense(*this, dense.data());
            // Matrix<CoefficientPrecision> U(rk - 1, size_s);
            // transpose(Ut, U);

            // // std::cerr << "aucune approximation trouvée , rk=" << rk << "sur un bloc de taille " << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << " et de norme : " << normFrob(dense) << std::endl;
            // std::cerr << "pas d'approximation ,  l'erreur a la fin est " << normFrob(dense - L * U) << std::endl;
            std::cout << "pas d'approximation " << std::endl;
        }
        return res;
    }

    CoefficientPrecision get_coeff(const int &i, const int &j) const {
        // Début et fin des colonnes pour la ligne i
        int row_start = row_csr[i];
        int row_end   = row_csr[i + 1];

        // Parcourir les colonnes associées à la ligne i
        for (int idx = row_start; idx < row_end; idx++) {
            if (col_csr[idx] == j) {
                // Si la colonne j est trouvée, retourner la valeur correspondante
                return data[idx];
            }
        }
        // Si la colonne j n'est pas trouvée, retourner 0.0
        return 0.0;
    }

    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                ptr[i + M * j] = this->get_coeff(i + row_offset, j + col_offset);
            }
        }
    }
};

// template <class CoefficientPrecision, class CoordinatePrecision>
// class COO_generator : public VirtualGenerator<CoefficientPrecision> {
//   private:
//     std::vector<double> data;
//     std::vector<double> row_coo;
//     std::vector<double> col_coo;
//     const Cluster<CoordinatePrecision> &target;
//     const Cluster<CoordinatePrecision> &source;

//   public:
//     COO_generator(const Cluster<CoordinatePrecision> &target0,
//                   const Cluster<CoordinatePrecision> &source0,
//                   const std::vector<double> &row_coo0,
//                   const std::vector<double> &col_coo0,
//                   const std::vector<double> &val_coo0) : target(target0), source(source0) {

//         int M     = target0.get_size(); // nombre de lignes après permutation
//         int N     = source0.get_size(); // nombre de colonnes après permutation
//         auto perm = target0.get_permutation();
//         row_coo.resize(row_coo0.size());
//         col_coo.resize(row_coo0.size());
//         data.resize(row_coo0.size());
//         for (int k = 0; k < row_coo.size(); ++k) {
//             int kperm  = perm[row_coo0[k]];
//             row_coo[k] = kperm;
//             int lperm  = perm[col_coo0[k]];
//             col_coo[k] = lperm;
//             data[k]    = val_coo0[k];
//         }
//     }

//     COO_generator(const Cluster<CoordinatePrecision> &target0,
//                   const Cluster<CoordinatePrecision> &source0,
//                   const std::vector<double> &row_csr0,
//                   const std::vector<double> &col_csr0,
//                   const std::vector<double> &val_csr0,
//                   const bool flag) : target(target0), source(source0) {

//         int M = target0.get_size(); // nombre de lignes après permutation
//         int N = source0.get_size(); // nombre de colonnes après permutation
//         std::vector<CoefficientPrecision> row_temp, col_temp;
//         for (int i = 0; i < M; ++i) {
//             int start = row_csr0[i];
//             int end   = row_csr0[i + 1];

//             // Ajouter les indices de ligne et de colonne, et la valeur
//             for (int j = start; j < end; ++j) {
//                 row_temp.push_back(i);           // Indice de ligne
//                 col_temp.push_back(col_csr0[j]); // Indice de colonne
//                 data.push_back(val_csr0[j]);     // Valeur non nulle correspondante
//             }
//         }
//         auto perm = target0.get_permutation();
//         col_coo.resize(M);
//         row_coo.resize(M);
//         for (int k = 0; k < row_coo.size(); ++k) {
//             int kperm  = perm[row_temp[k]];
//             row_coo[k] = kperm;
//             int lperm  = perm[col_coo0[k]];
//             col_coo[k] = lperm;
//             data[k]    = val_coo0[k];
//         }
//     }

//     void mat_vect(const std::vector<CoefficientPrecision> &in, std::vector<CoefficientPrecision> &out) {
//         int n = row_csr.size() - 1;
//         for (int i = 0; i < n; ++i) {
//             for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
//                 out[i] += data[j] * in[col_csr[j]];
//             }
//         }
//     }
//     // void mat_vect_bloc(const int &size_t, const int &size_s, const int &oft, const int &ofs, const std::vector<CoefficientPrecision> &in, std::vector<CoefficientPrecision> &out) {
//     //     for (int i = oft; i < size_t + oft; ++i) {
//     //         for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
//     //             int col = col_idx[j] - ofs;     // Ajuster l'indice de colonne avec ofs
//     //             if (col >= 0 && col < size_s) { // Vérifier si la colonne est dans le sous-bloc
//     //                 out[i] += data[j] * in[col];
//     //             }
//     //         }
//     //     }
//     // }

//     void mat_vect_bloc(const int &size_t, const int &size_s, const int &oft, const int &ofs, const std::vector<CoefficientPrecision> &in, std::vector<CoefficientPrecision> &out, char trans) const {
//         if (trans == 'N') {
//             // Multiplication standard: CSR * vecteur
//             for (int i = oft; i < size_t + oft; ++i) {
//                 for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
//                     int col = col_csr[j] - ofs;     // Ajuster l'indice de colonne avec ofs
//                     if (col >= 0 && col < size_s) { // Vérifier si la colonne est dans le sous-bloc
//                         out[i] += data[j] * in[col];
//                     }
//                 }
//             }
//         } else if (trans == 'T') {
//             // Multiplication transposée: CSR^T * vecteur
//             // Note: Pour la transposée, les rôles de `i` et `col` sont inversés.
//             for (int i = oft; i < size_t + oft; ++i) {
//                 for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
//                     int col = col_csr[j] - ofs;            // Ajuster l'indice de colonne avec ofs
//                     if (col >= 0 && col < size_s) {        // Vérifier si la colonne est dans le sous-bloc
//                         out[col] += data[j] * in[i - oft]; // Transposer le calcul : out[col] = A^T * in
//                     }
//                 }
//             }
//         }
//     }

//     std::vector<Matrix<CoefficientPrecision>> rd_ACA(const int &size_t, const int &size_s, const int &oft, const int &ofs, const double &tolerance, bool &flagout) const {
//         int rk = 1;
//         Matrix<CoefficientPrecision> L(size_t, 1);
//         Matrix<CoefficientPrecision> Ut(size_s, 1);
//         Matrix<CoefficientPrecision> temp(size_t, size_t);
//         std::vector<Matrix<CoefficientPrecision>> res;
//         for (int k = 0; k < size_t; ++k) {
//             temp(k, k) = 1.0;
//         }
//         double norme_LU = 0.0;
//         // auto bloc       = this->copy_submatrix(size_t, size_s, oft, ofs);
//         auto w = gaussian_vector(size_s, 0.0, 1.0);
//         while ((rk < size_t / 2) && (rk < size_s / 2)) {
//             std::vector<CoefficientPrecision> Aw(size_t, 0.0);
//             // bloc->add_vector_product('N', 1.0, w.data(), 1.0, Aw.data());
//             this->mat_vect_bloc(size_t, size_s, oft, ofs, w, Aw, 'N');
//             std::vector<CoefficientPrecision> lr(size_t, 0.0);
//             temp.add_vector_product('N', 1.0, Aw.data(), 1.0, lr.data());
//             auto norm_lr = norm2(lr);
//             if (norm_lr < 1e-20) {
//                 break;
//             } else {
//                 lr = mult(1.0 / norm_lr, lr);
//             }
//             std::vector<CoefficientPrecision> ur(size_s, 0.0);
//             // bloc->add_vector_product('T', 1.0, lr.data(), 1.0, ur.data());
//             this->mat_vect_bloc(size_t, size_s, oft, ofs, lr, ur, 'T');
//             auto norm_ur = norm2(ur);
//             // std::cout << "norm2 ur : " << norm_ur << std::endl;
//             Matrix<CoefficientPrecision> lr_ur(size_t, size_t);
//             double alpha = 1.0;
//             double beta  = -1.0;
//             int kk       = 1;
//             Blas<CoefficientPrecision>::gemm("N", "T", &size_t, &size_t, &kk, &alpha, lr.data(), &size_t, ur.data(), &size_t, &alpha, lr_ur.data(), &size_t);
//             std::vector<CoefficientPrecision> Llr(L.nb_cols(), 0.0);
//             L.add_vector_product('T', 1.0, lr.data(), 1.0, Llr.data());
//             std::vector<CoefficientPrecision> ULlr(size_s, 0.0);
//             Ut.add_vector_product('N', 1.0, Llr.data(), 1.0, ULlr.data());
//             double trace = 0.0;
//             for (int k = 0; k < std::min(size_t, size_s); ++k) {
//                 trace += (ULlr[k] * ur[k]);
//             }
//             auto nr_lrur = normFrob(lr_ur);
//             auto nr      = norme_LU + std::pow(nr_lrur, 2.0) + 2 * trace;
//             if (rk > 1 && (std::pow(norm_lr * norm_ur * (size_s - rk), 2.0) <= std::pow(tolerance, 2.0) * nr)) {
//                 break;
//             }
//             if (rk == 1) {
//                 std::copy(lr.data(), lr.data() + size_t, L.data());
//                 std::copy(ur.data(), ur.data() + size_s, Ut.data());
//                 // std::cout << "!!!!!! aprés affectation " << normFrob(L) << ',' << normFrob(Ut) << "! " << norm_lr << ',' << norm_ur << std::endl;
//             } else {
//                 Matrix<CoefficientPrecision> new_L(size_t, rk);
//                 std::copy(L.data(), L.data() + L.nb_rows() * L.nb_cols(), new_L.data());
//                 std::copy(lr.data(), lr.data() + size_t, new_L.data() + (rk - 1) * size_t);
//                 Matrix<CoefficientPrecision> new_U(size_s, rk);
//                 std::copy(Ut.data(), Ut.data() + Ut.nb_rows() * Ut.nb_cols(), new_U.data());
//                 std::copy(ur.data(), ur.data() + size_s, new_U.data() + (rk - 1) * size_s);
//                 L  = new_L;
//                 Ut = new_U;
//             }
//             w = gaussian_vector(size_s, 0.0, 1.0);
//             Blas<CoefficientPrecision>::gemm("N", "T", &size_t, &size_t, &kk, &beta, lr.data(), &size_t, lr.data(), &size_t, &alpha, temp.data(), &size_t);
//             norme_LU += std::pow(nr_lrur, 2.0);
//             rk += 1;
//         }
//         if (rk >= std::min(size_t / 2, size_s / 2)) {
//             flagout = false;
//         }
//         if (flagout) {
//             if (rk == 1) {
//                 Matrix<CoefficientPrecision> U(1, size_s);
//                 res.push_back(L);
//                 res.push_back(U);
//             } else {
//                 Matrix<CoefficientPrecision> U(rk - 1, size_s);
//                 transpose(Ut, U);
//                 // LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_mat(L, U);
//                 // bloc->delete_children();
//                 // bloc->set_low_rank_data(lr_mat);
//                 res.push_back(L);
//                 res.push_back(U);
//             }
//         } else {
//             // Matrix<CoefficientPrecision> dense(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
//             // copy_to_dense(*this, dense.data());
//             // Matrix<CoefficientPrecision> U(rk - 1, size_s);
//             // transpose(Ut, U);

//             // // std::cerr << "aucune approximation trouvée , rk=" << rk << "sur un bloc de taille " << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << " et de norme : " << normFrob(dense) << std::endl;
//             // std::cerr << "pas d'approximation ,  l'erreur a la fin est " << normFrob(dense - L * U) << std::endl;
//             std::cout << "pas d'approximation " << std::endl;
//         }
//         return res;
//     }

//     CoefficientPrecision get_coeff(const int &i, const int &j) const {
//         // Début et fin des colonnes pour la ligne i
//         int row_start = row_csr[i];
//         int row_end   = row_csr[i + 1];

//         // Parcourir les colonnes associées à la ligne i
//         for (int idx = row_start; idx < row_end; idx++) {
//             if (col_csr[idx] == j) {
//                 // Si la colonne j est trouvée, retourner la valeur correspondante
//                 return data[idx];
//             }
//         }
//         // Si la colonne j n'est pas trouvée, retourner 0.0
//         return 0.0;
//     }

//     void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
//         const auto &target_permutation = target.get_permutation();
//         const auto &source_permutation = source.get_permutation();
//         for (int i = 0; i < M; ++i) {
//             for (int j = 0; j < N; ++j) {
//                 ptr[i + M * j] = this->get_coeff(i + row_offset, j + col_offset);
//             }
//         }
//     }
// };

template <class CoefficientPrecision, class CoordinatePrecision>
void build_hmatrix(const CSR_generator<double, double> &generator, HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix) {
    auto &t  = hmatrix->get_target_cluster();
    auto &s  = hmatrix->get_source_cluster();
    bool adm = hmatrix->compute_admissibility(t, s);
    if (adm) {
        bool flag = true;
        auto res  = generator.rd_ACA(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), hmatrix->get_epsilon(), flag);
        if (flag) {
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(res[0], res[1]);
            hmatrix->set_low_rank_data(lr);
        } else {
            if (t.get_children().size() > 0) {
                for (auto &t_child : t.get_children()) {
                    for (auto &s_child : s.get_children()) {
                        auto child = hmatrix->add_child(t_child.get(), s_child.get());
                        build_hmatrix(generator, child);
                    }
                }
            } else {
                Matrix<CoefficientPrecision> dense(t.get_size(), s.get_size());
                generator.copy_submatrix(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), dense.data());
                hmatrix->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(dense));
            }
        }
    } else {
        if (t.get_children().size() > 0) {
            for (auto &t_child : t.get_children()) {
                for (auto &s_child : s.get_children()) {
                    auto child = hmatrix->add_child(t_child.get(), s_child.get());
                    build_hmatrix(generator, child);
                }
            }
        } else {
            Matrix<CoefficientPrecision> dense(t.get_size(), s.get_size());
            generator.copy_submatrix(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), dense.data());
            hmatrix->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(dense));
        }
    }
}
double random_value() {
    // static std::random_device rd;                     // Graine aléatoire
    // static std::mt19937 gen(rd());                    // Mersenne Twister comme générateur
    // static std::normal_distribution<> dist(0.0, 1.0); // Distribution normale

    // Retourner un nombre aléatoire suivant N(0,1)
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    double number = distribution(generator);
    return number;
}
template <class CoefficientPrecision, class CoordinatePrecision>
class CSR_generator_random : public VirtualGenerator<CoefficientPrecision> {
  private:
    std::vector<double> data;
    std::vector<double> row_csr;
    std::vector<double> col_csr;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;
    const std::vector<CoordinatePrecision> &points;
    const int dim;

  public:
    // CSR_generator(std::vector<double> &data0, std::vector<double> &row_csr0, std::vector<double> &col_csr0, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0) : data(data0), row_csr(row_csr0), col_csr(col_csr0), target(target0), source(source0) {}

    CSR_generator_random(const Cluster<CoordinatePrecision> &target0,
                         const Cluster<CoordinatePrecision> &source0,
                         const std::vector<double> &row_csr0,
                         const std::vector<double> &col_csr0,
                         const std::vector<double> &val_csr0,
                         const std::vector<CoordinatePrecision> &points0,
                         const int &dim0) : target(target0), source(source0), points(points0), dim(dim0) {

        int M = target0.get_size(); // nombre de lignes après permutation
        int N = source0.get_size(); // nombre de colonnes après permutation

        // Initialisation des nouvelles structures CSR
        row_csr.resize(M + 1, 0); // row_csr_out contient les décalages pour chaque ligne
        int rep = 0;
        // Boucle sur les lignes permutées
        for (int i = 0; i < M; i++) {
            int row_perm = target0.get_permutation()[i];
            int r1       = row_csr0[row_perm];     // début de la ligne dans row_csr
            int r2       = row_csr0[row_perm + 1]; // fin de la ligne dans row_csr

            // Boucle sur les colonnes permutées pour la ligne courante
            int flag = 0;
            for (int j = 0; j < N; j++) {
                int col_perm = source0.get_permutation()[j];

                // Extraire les colonnes et les valeurs de la ligne permutée
                std::vector<int> row_iperm(r2 - r1);
                std::copy(col_csr0.begin() + r1, col_csr0.begin() + r2, row_iperm.begin());

                // Rechercher l'indice de la colonne permutée
                auto it = std::find(row_iperm.begin(), row_iperm.end(), col_perm);
                if (it != row_iperm.end()) {
                    int idx = std::distance(row_iperm.begin(), it);
                    col_csr.push_back(j);               // colonne après permutation
                    data.push_back(val_csr0[r1 + idx]); // valeur associée
                    flag += 1;
                }
            }
            row_csr[i + 1] = row_csr[i] + flag;
        }
    }
    CoefficientPrecision get_coeff(const int &i, const int &j) const {
        // Début et fin des colonnes pour la ligne i
        int row_start = row_csr[i];
        int row_end   = row_csr[i + 1];

        // Parcourir les colonnes associées à la ligne i
        for (int idx = row_start; idx < row_end; idx++) {
            if (col_csr[idx] == j) {
                // Si la colonne j est trouvée, retourner la valeur correspondante
                return data[idx];
            }
        }
        // std::cout << "radom =";
        auto rd = random_value();
        // std::vector<CoefficientPrecision> p1(dim);
        // std::vector<CoefficientPrecision> p2(dim);
        // int iperm = target.get_permutation()[i];
        // int jperm = source.get_permutation()[j];
        // for (int k = 0; k < dim; ++k) {
        //     p1[k] = points[dim * iperm + k];
        //     p2[k] = points[dim * jperm + k];
        // }
        // auto nr = 1.0 * rd / (1.0 + norm2(p1 - p2));
        auto nr = 1.0 * rd / std::pow(std::abs(i - j), 2.0);
        return nr;
    }

    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        // std::cout << "begin copy_sub" << std::endl;
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                ptr[i + M * j] = this->get_coeff(i + row_offset, j + col_offset);
            }
        }
        // std::cout << "end copy sub" << std::endl;
    }
};
template <class CoefficientPrecision, class CoordinatePrecision>
class my_generator : public VirtualGenerator<CoefficientPrecision> {
  private:
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;
    const std::vector<CoordinatePrecision> points;

  public:
    my_generator(const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0, const std::vector<CoordinatePrecision> &points0) : target(target0), source(source0), points(points0) {}
    CoefficientPrecision get_coeff(const int &i, const int &j) const {
        int iperm = target.get_permutation()[i];
        int jperm = source.get_permutation()[j];
        auto nr   = (points[3 * iperm] - points[3 * jperm]) * (points[3 * iperm] - points[3 * jperm]) + (points[3 * iperm + 1] - points[3 * jperm + 1]) * (points[3 * iperm + 1] - points[3 * jperm + 1]) + (points[3 * iperm + 2] - points[3 * jperm + 2]) * (points[3 * iperm + 2] - points[3 * jperm + 2]);
        return 1.0 / (0.1 + nr);
    }
    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                ptr[i + M * j] = this->get_coeff(i + row_offset, j + col_offset);
            }
        }
    }
};

template <class CoefficientPrecision, class CoordinatePrecision>
class Matricegenerator : public VirtualGenerator<CoefficientPrecision> {
  private:
    Matrix<CoefficientPrecision> mat;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;

  public:
    Matricegenerator(Matrix<CoefficientPrecision> &mat0, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0) : mat(mat0), target(target0), source(source0) {}

    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = mat(target_permutation[i + row_offset], source_permutation[j + col_offset]);
            }
        }
    }

    Matrix<CoefficientPrecision> get_mat() { return mat; }

    Matrix<CoefficientPrecision> get_perm_mat() {
        CoefficientPrecision *ptr = new CoefficientPrecision[target.get_size() * source.get_size()];
        this->copy_submatrix(target.get_size(), target.get_size(), 0, 0, ptr);
        Matrix<CoefficientPrecision> M;
        M.assign(target.get_size(), source.get_size(), ptr, true);
        return M;
    }
    Matrix<CoefficientPrecision> get_unperm_mat() {
        CoefficientPrecision *ptr      = new CoefficientPrecision[target.get_size() * source.get_size()];
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < target.get_size(); i++) {
            for (int j = 0; j < source.get_size(); j++) {
                ptr[target_permutation[i] + source_permutation[j] * target.get_size()] = mat(i, j);
            }
        }
        Matrix<CoefficientPrecision> M;
        M.assign(target.get_size(), source.get_size(), ptr, true);
        return M;
    }
};
template <class CoefficientPrecision, class CoordinatePrecision>
Matrix<CoefficientPrecision> get_unperm_mat(const Matrix<CoefficientPrecision> &mat, Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source) {
    CoefficientPrecision *ptr      = new CoefficientPrecision[target.get_size() * source.get_size()];
    const auto &target_permutation = target.get_permutation();
    const auto &source_permutation = source.get_permutation();
    for (int i = 0; i < target.get_size(); i++) {
        for (int j = 0; j < source.get_size(); j++) {
            ptr[target_permutation[i] + source_permutation[j] * target.get_size()] = mat(i, j);
        }
    }
    Matrix<CoefficientPrecision> M;
    M.assign(target.get_size(), source.get_size(), ptr, false);
    return M;
}
template <class CoefficientPrecision, class CoordinatePrecision>
class Matricegenerator_noperm : public VirtualGenerator<CoefficientPrecision> {
  private:
    Matrix<CoefficientPrecision> mat;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;

  public:
    Matricegenerator_noperm(Matrix<CoefficientPrecision> &mat0, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0) : mat(mat0), target(target0), source(source0) {}

    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = mat(i, j);
            }
        }
    }

    // Matrix<CoefficientPrecision> get_mat() { return mat; }

    // Matrix<CoefficientPrecision> get_perm_mat() {
    //     CoefficientPrecision *ptr = new CoefficientPrecision[target.get_size() * source.get_size()];
    //     this->copy_submatrix(target.get_size(), target.get_size(), 0, 0, ptr);
    //     Matrix<CoefficientPrecision> M;
    //     M.assign(target.get_size(), source.get_size(), ptr, true);
    //     return M;
    // }
    // Matrix<CoefficientPrecision> get_unperm_mat() {
    //     CoefficientPrecision *ptr      = new CoefficientPrecision[target.get_size() * source.get_size()];
    //     const auto &target_permutation = target.get_permutation();
    //     const auto &source_permutation = source.get_permutation();
    //     for (int i = 0; i < target.get_size(); i++) {
    //         for (int j = 0; j < source.get_size(); j++) {
    //             ptr[target_permutation[i] + source_permutation[j] * target.get_size()] = mat(i, j);
    //         }
    //     }
    //     Matrix<CoefficientPrecision> M;
    //     M.assign(target.get_size(), source.get_size(), ptr, true);
    //     return M;
    // }
};

Matrix<double> extract(const int &nr, const int &nc, const int &ofr, const int &ofc, Matrix<double> M) {
    Matrix<double> res(nr, nc);
    for (int k = 0; k < nr; ++k) {
        for (int l = 0; l < nc; ++l) {
            res(k, l) = M(k + ofr, l + ofc);
        }
    }
    return res;
}
// Champ de vecteur pour nos test
template <class CoordinatePrecision>
std::vector<CoordinatePrecision> my_vector(const std::vector<CoordinatePrecision> X) {
    std::vector<CoordinatePrecision> res(X.size());
    // double nr  = std::pow(1.0 + (600.0 - X[0]) * (600.0 - X[0]) / 90000.0, 0.5);

    // res[0] = 1.0;
    // res[1] = (600.0 - X[0]) / (300.0);
    double alpha = 1.0;
    res[0]       = 1.0;
    res[1]       = sin(3.14 * alpha * X[0] / 1200.0);
    return res;
}

template <class CoordinatePrecision>
std::vector<CoordinatePrecision> transform_point(std::vector<CoordinatePrecision> (*func)(std::vector<CoordinatePrecision>), const std::vector<CoordinatePrecision> &x0, const CoordinatePrecision &xmin, double &delta) {
    auto current_point = x0;
    auto dist          = 0.0;
    // std::cout << "________________________________" << std::endl;
    while (xmin <= current_point[0]) {
        // auto next_point = current_point - delta * func(current_point);
        std::vector<CoordinatePrecision> vect = func(current_point);
        // std::cout << current_point[0] << ',' << current_point[1] << '|' << vect[0] << ',' << vect[1] << std::endl;
        // double nr                             = 0;
        // for (auto &v : vect) {
        //     nr += (v * v);
        // }
        // auto n = std::pow(1.0 * nr, 0.5);
        // auto vect_norm = vect / norm2(vect);
        // std::cout << "vect :  " << vect[0] << ',' << vect[1] << " norme :" << norm2(vect) << " norme vect : " << vect_norm[0] << ',' << vect_norm[1] << std::endl;
        // std::cout << "norme de v " << n << std::endl;
        // for (int k = 0; k < vect.size(); ++k) {
        //     vect[k] = vect[k] / (1.0 * n);
        // }
        // auto next_point = current_point - multiply_vector_by_scalar(vect_norm, delta * 1.0);
        // auto next_x = current_point[0] - delta * vect[0];
        // auto next_y = current_point[1] - delta * vect[1];
        // // auto next_point = current_point - multiply_vector_by_scalar(vect, delta * 1.0);
        // std::vector<CoordinatePrecision> next_point(2);
        // next_point[0] = next_x;
        // next_point[1] = next_y;
        // auto next_point = current_point - multiply_vector_by_scalar(vect, delta);
        auto next_point = current_point - mult(delta, vect);

        dist += delta;
        // std::cout << "next point : " << next_point[0] << ',' << next_point[1] << std::endl;
        current_point = next_point;
    }
    double scale = 1.0;
    std::vector<CoordinatePrecision> res(current_point);
    res[0] = scale * dist;
    res[1] = current_point[1];
    return res;
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

template <class CoordinatePrecision>
std::vector<std::vector<CoordinatePrecision>> trajectoire(std::vector<CoordinatePrecision> (*func)(std::vector<CoordinatePrecision>), const std::vector<CoordinatePrecision> &x0, const CoordinatePrecision &xmin, double &delta) {
    auto current_point = x0;
    auto dist          = 0.0;
    std::vector<std::vector<CoordinatePrecision>> traj;
    traj.push_back(x0);
    // auto x = current_point[0];
    // auto y = current_point[1];
    while (current_point[0] >= xmin) {
        // auto next_point = current_point - delta * func(current_point);
        std::vector<CoordinatePrecision> vect = func(current_point);
        // double nr                             = 0;
        // for (auto &v : vect) {
        //     nr += (v * v);
        // }
        // auto n = std::pow(1.0 * nr, 0.5);
        auto vect_norm = vect / norm2(vect);
        // std::cout << "norme de v " << n << std::endl;
        // for (int k = 0; k < vect.size(); ++k) {
        //     vect[k] = vect[k] / (1.0 * n);
        // }
        // auto next_point = current_point - multiply_vector_by_scalar(vect_norm, delta * 1.0);
        auto next_point = current_point - multiply_vector_by_scalar(vect, delta);
        traj.push_back(next_point);
        dist += delta;
        // std::cout << "next point : " << next_point[0] << ',' << next_point[1] << std::endl;
        current_point = next_point;
    }
    return traj;
}

template <class CoordinatePrecision>
std::vector<CoordinatePrecision> transform_cluster(std::vector<CoordinatePrecision> (*func)(std::vector<CoordinatePrecision>), const std::vector<CoordinatePrecision> &points, const CoordinatePrecision &xmin, double &delta) {
    // std::vector<std::vector<CoordinatePrecision>> res(points.size());
    std::vector<CoordinatePrecision> res(points.size());
    // for (auto &x : points) {
    for (int k = 0; k < points.size() / 2; ++k) {
        std::vector<CoordinatePrecision> x(2);
        x[0]       = points[2 * k];
        x[1]       = points[2 * k + 1];
        auto new_x = transform_point(func, x, xmin, delta);
        // res[rep] = new_x ;
        res[2 * k]     = new_x[0];
        res[2 * k + 1] = new_x[1];
    }
    return res;
}

void save_points_to_csv(const std::vector<std::vector<double>> &points, const std::string &filename) {
    // Ouvrir le fichier en mode écriture
    std::ofstream file(filename);

    // Vérifier que le fichier est ouvert correctement
    if (!file.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir le fichier " << filename << std::endl;
        return;
    }

    // Parcourir chaque point (vecteur de doubles)
    for (const auto &point : points) {
        // Parcourir les coordonnées d'un point
        for (size_t i = 0; i < point.size(); ++i) {
            file << point[i];
            if (i < point.size() - 1) {
                file << ","; // Ajouter une virgule entre les coordonnées
            }
        }
        file << "\n"; // Nouvelle ligne après chaque point
    }

    // Fermer le fichier
    file.close();
    std::cout << "Les points ont été sauvegardés dans le fichier " << filename << std::endl;
}

template <class CoordinatePrecision>
std::vector<std::vector<CoordinatePrecision>> generate_grid(const int N, const int space_dim, const std::vector<std::pair<CoordinatePrecision, CoordinatePrecision>> &limits) {
    std::vector<std::vector<CoordinatePrecision>> res(std::pow(N * 1.0, space_dim * 1.0));
    if (space_dim == 2) {
        auto xm = limits[0].first;
        auto xM = limits[0].second;
        auto ym = limits[0].first;
        auto yM = limits[0].second;
        for (int k = 0; k < N; ++k) {
            for (int l = 0; l < N; ++l) {
                std::vector<CoordinatePrecision> point(2);
                auto xx        = xm + k * (xM - xm) * 1.0 / (N * 1.0);
                auto yy        = ym + l * (yM - ym) * 1.0 / (N * 1.0);
                point[0]       = xx;
                point[1]       = yy;
                res[k + N * l] = point;
            }
        }
    } else {
        std::cerr << "mauvaise space dim" << std::endl;
    }
    return res;
}
int main() {
    // TESTS POUR PYTHON POUR VERIFIER QUE LE REDRESSEMENT MARCHE
    // double limit0 = 0.0;
    // double limit1 = 5.0;
    // int N         = 20;
    // int space_dim = 2;
    // double delta  = 0.5;
    // std::pair<double, double> l;
    // l.first  = limit0;
    // l.second = limit1;
    // std::vector<std::pair<double, double>> lim;
    // lim.push_back(l);
    // lim.push_back(l);
    // auto points = generate_grid(N, space_dim, lim);
    // save_points_to_csv(points, "grid_points.csv");
    // auto deform_points = transform_cluster(my_vector, points, limit0, delta);
    // save_points_to_csv(deform_points, "grid_deformed.csv");
    // auto traj = trajectoire(my_vector, points[points.size() - 1], limit0, delta);
    // save_points_to_csv(traj, "trajectoire.csv");
    //////////////////////////////

    /// TEST AVEC HTOOL ET LES MATRICES DE LEO
    for (int ss = 2; ss < 3; ++ss) {
        double epsilon = 1e-6;
        double eta     = 2.0;
        // int sizex      = 100;
        // int sizey      = 100;
        int sizex = 50 * ss;
        int sizey = 50 * ss;
        int size  = sizex * sizey;

        std::cout << "epsilon et eta : " << epsilon << ',' << eta << std::endl;
        std::cout << "_______________________________________________________" << std::endl;
        std::cout << "sizex x sizey  = " << sizex << " x " << sizey << std::endl;
        // auto rd = random_value();
        // std::cout << "random : " << rd << std::endl;
        // ;

        ///////////////////////
        ////// GENERATION MAILLAGE  :  size points en 2 dimensions

        std::vector<double> p(2 * size);
        // std::vector<double> p3(3 * size);

        for (int j = 0; j < sizex; j++) {
            for (int k = 0; k < sizey; k++) {

                p[2 * (j + k * sizex) + 0] = j * 1200 / sizex;
                p[2 * (j + k * sizex) + 1] = k * 1200 / sizey;
            }
        }
        // create_disk(3, 0.0, size, p3.data());

        // Creation clusters
        // normal
        /////// MATRICE HIERARCHIQUE POUR ESAYER DE FAIRE DES ACA SUR LES BLOCS
        // std::vector<double> p1(3 * size);
        // std::cout << p1.size() << ',' << size << std::endl;
        // create_disk(3, 0.0, size, p1.data());

        // ClusterTreeBuilder<double> recursive_build_strategy_1;
        // std::shared_ptr<Cluster<double>> root_cluster = std::make_shared<Cluster<double>>(recursive_build_strategy_1.create_cluster_tree(p1.size() / 3, 3, p1.data(), 2, 2));
        // Matrix<double> reference(size, size);
        // my_generator<double, double> generator(*root_cluster, *root_cluster, p1);

        // // for (int k = 0; k < size; ++k) {
        // //     for (int l = 0; l < size; ++l) {
        // //         reference(k, l) = generator.get_coef(k, l);
        // //     }
        // // }
        // // std::cout << "norme reference : " << normFrob(reference) << std::endl;
        // generator.copy_submatrix(size, size, 0, 0, reference.data());
        // std::cout << "norme generator: " << normFrob(reference) << std::endl;

        // HMatrixTreeBuilder<double, double> hmatrix_tree_builder(*root_cluster, *root_cluster, epsilon, 0, 'N', 'N', -1, -1, -1);

        // auto root_hmatrix = hmatrix_tree_builder.build(generator);
        // Matrix<double> root_dense(size, size);
        // copy_to_dense(root_hmatrix, root_dense.data());
        // std::cout << " erreur assemblage : " << normFrob(root_dense - reference) / normFrob(reference) << std::endl;
        // std::cout << "compression " << root_hmatrix.get_compression() << std::endl;
        // ACA SUR UN BLOC
        // for (auto &l : root_hmatrix.get_leaves()) {
        //     if (l->get_target_cluster().get_size() > 100) {
        //         std::cout << l->get_target_cluster().get_size() << ',' << l->get_source_cluster().get_size() << ',' << l->get_target_cluster().get_offset() << ',' << l->get_source_cluster().get_offset() << std::endl;
        //     }
        // }
        // for (auto &l : root_hmatrix.get_children()[1]->get_children()[1]->get_children()[1]->get_children()[1]->get_children()[1]->get_children()) {
        //     std::cout << l->get_target_cluster().get_size() << ',' << l->get_source_cluster().get_size() << ',' << l->get_target_cluster().get_offset() << ',' << l->get_source_cluster().get_offset() << std::endl;
        // }
        // auto bloc = root_hmatrix.get_block(156, 157, 156, 9843);
        // Matrix<double> bloc_dense(156, 157);
        // copy_to_dense(*bloc, bloc_dense.data());
        // bool flag = true;
        // root_hmatrix.random_ACA(156, 157, 156, 9843, 1e-6, flag);
        // if (flag) {
        //     // std::cout<< "apparemment il s'en"reference_dense
        //     Matrix<double> uv = root_hmatrix.get_block(156, 157, 156, 9843)->get_low_rank_data()->get_U() * root_hmatrix.get_block(156, 157, 156, 9843)->get_low_rank_data()->get_V();
        //     std::cout << " erreur ACA sur le bloc : " << normFrob(bloc_dense - uv) / normFrob(bloc_dense) << std::endl;
        // }$
        /// ACA HIERARCHIQUE
        // root_hmatrix.set_eta(eta);
        // Matrix<double> root_aca(size, size);
        // root_hmatrix.iterate_ACA(1e-6);
        // root_hmatrix.iterate_ACA(*root_cluster, *root_cluster, 1e-6);
        // HMatrix<double, double> test_aca(root_cluster, root_cluster);
        // root_hmatrix.format_ACA(&test_aca, 1e-6);
        // std::cout << "ACA ok " << std::endl;

        // copy_to_dense(test_aca, root_aca.data());
        // std::cout << "erreur aprés ACA : " << normFrob(reference - root_aca) / normFrob(reference) << std::endl;
        // std::cout << "compression aca " << test_aca.get_compression() << std::endl;

        ///////////
        // auto bloc = root_hmatrix.get_block(156, 157, 156, 9843);
        // Matrix<double> dense_bloc(156, 157);
        // std::cout << bloc->get_target_cluster().get_size() << ',' << bloc->get_source_cluster().get_size() << ',' << bloc->get_target_cluster().get_offset() << ',' << bloc->get_source_cluster().get_offset() << std::endl;
        // copy_to_dense(*bloc, dense_bloc.data());
        // auto LU = random_ACA(dense_bloc, epsilon, false);
        // if (false == true) {
        //     std::cout << "erreur ACA sur matrices : " << normFrob(LU[0] * LU[1] - dense_bloc) / normFrob(dense_bloc) << std::endl;
        // }
        // std::cout << "compression du sous boc avant ACA " << bloc->get_compression() << std::endl;
        // bool flag = false;
        // Matrix<double> L, U;
        // // root_hmatrix.random_ACA(156, 157, 156, 9843, L, U, epsilon, flag);
        // root_hmatrix.random_ACA(156, 157, 1538.93flag << std::endl;
        // if (flag) {
        //     std::cout << "rang de l'approximation : " << L.nb_cols() << std::endl;
        //     std::cout << "erreur " << normFrob(L * U - dense_bloc) / normFrob(dense_bloc) << std::endl;
        // }

        // // Recuperation du CSR
        for (int Eps = 2; Eps < 3; ++Eps) {
            // save_cluster_tree(*root_cluster, "root_cluster");
            int eps = 0;
            // int eps = 2 * Eps;
            std::cout << "______________________________" << std::endl;
            std::cout << "eps = " << eps << std::endl;
            // auto csr_file = get_csr("/work/sauniear/Documents/Code_leo/VersionLight2/alpha_1v0_epsilon_" + std::to_string(eps) + ".csv");
            // auto csr_file = get_csr("/work/sauniear/Documents/Code_leo/VersionLight2/size_" + std::to_string(sizex) + "x" + std::to_string(sizey) + "alpha1_epsilon" + std::to_string(eps) + ".csv");
            auto csr_file = get_csr("/work/sauniear/Documents/Code_leo/VersionLight2/size_" + std::to_string(sizex) + "x" + std::to_string(sizey) + "_alpha1_epsilon" + std::to_string(eps) + ".csv");

            auto val = csr_file[0];
            auto row = csr_file[1];
            auto col = csr_file[2];
            std::cout << "csr size : " << row.size() << ',' << col.size() << "     ,     " << val.size() << std::endl;
            // Matrix<double> vrai_mat(size, size);
            // for (int i = 0; i < size; ++i) {
            //     for (int j = row[i]; j < row[i + 1]; ++j) {
            //         vrai_mat(i, col[j]) = val[j];
            //     }
            // }

            // ///////////////////////////////////////////////////////////////////////////////////
            // /// on fait un LU pour récuperer l'inverse
            // auto reference_dense = vrai_mat;
            // std::vector<int> ipiv(size, 0.0);
            // int info = -1;
            // Lapack<double>::getrf(&size, &size, vrai_mat.data(), &size, ipiv.data(), &info);
            // Matrix<double> L(size, size);
            // Matrix<double> U(size, size);
            // for (int k = 0; k < size; ++k) {
            //     U(k, k) = vrai_mat(k, k);
            //     L(k, k) = 1.0;
            //     for (int l = 0; l < k; ++l) {
            //         L(k, l) = vrai_mat(k, l);
            //         U(l, k) = vrai_mat(l, k);
            //     }
            // }
            // auto LU = vrai_mat;
            // std::cout << " erreur LU : " << normFrob(L * U - reference_dense) / normFrob(reference_dense) << std::endl;
            // std::cout << "norm lu " << normFrob(LU) << std::endl;
            // std::vector<int> ipivo(size);+std::to_string(sizex)+
            // int lwork = size * size;
            // std::vector<double> work(lwork);
            // // Effectue l'inversion
            // Lapack<double>::getri(&size, vrai_mat.data(), &size, ipiv.data(), work.data(), &lwork, &info);

            ///////////////////////////////////////////////////////////////////////////////////

            //// MAILLAGE DEFORME
            double limit0 = 0.0;
            double delta  = 1.0;

            std::vector<double> p_deformed = transform_cluster(my_vector, p, limit0, delta);
            std::vector<std::vector<double>> deformed;
            std::vector<double> def;
            std::vector<double> points_ref;
            for (int k = 0; k < size; ++k) {
                std::vector<double> poin(2);
                poin[0]   = p_deformed[2 * k];
                poin[1]   = p_deformed[2 * k + 1];
                double zz = 0.0;
                deformed.push_back(poin);
                def.push_back(p_deformed[2 * k]);
                def.push_back(p_deformed[2 * k + 1]);
                def.push_back(zz);
            }
            std::cout << "pdeformed ok " << std::endl;
            // Creation clusters bande
            std::vector<double> bb(3);
            bb[0] = 1;
            bb[1] = 0;
            std::vector<double> bperp(3);
            bperp[0] = 0;
            bperp[1] = 1;
            ClusterTreeBuilder<double> directional_build;
            directional_build.set_minclustersize(20);
            // CHOIX DU SPLITTING
            std::shared_ptr<ConstantDirection<double>> strategy_directional = std::make_shared<ConstantDirection<double>>(bperp);
            // a commenter si on veut un splitting normal
            directional_build.set_direction_computation_strategy(strategy_directional);
            std::cout << "build cluster tree " << std::endl;
            std::shared_ptr<Cluster<double>> root_directional = std::make_shared<Cluster<double>>(directional_build.create_cluster_tree(def.size() / 3, 3, def.data(), 2, 2));
            std::cout << "cluster tree ok " << std::endl;
            // std::shared_ptr<Cluster<double>> root_directional = std::make_shared<Cluster<double>>(directional_build.create_cluster_tree(p_deformed.size() / 2, 2, p_deformed.data(), 2, 2));
            // save_clustered_geometry(*root_directional, 3, def.data(), "cluster_deformed", {0, 1, 2, 3});
            // save_points_to_csv(deformed, "deformed_test.csv");

            // HMATRIX BUILDER
            HMatrixTreeBuilder<double, double> hmatrix_directional_builder(*root_directional, *root_directional, epsilon, eta, 'N', 'N', -1, -1, -1);
            // HMatrixTreeBuilder<double, double> hmatrix_directional_lu_builder(*root_directional, *root_directional, epsilon, eta, 'N', 'N', -1, -1, -1);

            // HMatrixTreeBuilder<double, double> hmatrix_directional_builder(*root_directional, *root_directional, epsilon, eta, 'N', 'N', -1, -1, -1);
            // HMatrixTreeBuilder<double, double> hmatrix_directional_builder_inverse(*root_directional, *root_directional, epsilon, eta, 'N', 'N', -1, -1, -1);

            // CONDITION D ADMISSIBILITE
            // std::shared_ptr<Bande2D<double>> directional_admissibility = std::make_shared<Bande2D<double>>(bb, p_deformed);
            // std::shared_ptr<Bande2D<double>> directional_admissibility = std::make_shared<Bande2D<double>>(bb, def);

            // std::shared_ptr<Bande<double>> directional_admissibility = std::make_shared<Bande<double>>(bb, root_directional->get_permutation());
            // std::shared_ptr<Bandee<double>> directional_admissibility = std::make_shared<Bandee<double>>(bb, p_deformed);
            std::shared_ptr<Bandee<double>> directional_admissibility = std::make_shared<Bandee<double>>(bb, def);
            // std::shared_ptr<Bandee<double>> directional_admissibility = std::make_shared<Bandee<double>>(bb, points_ref);

            // a commenter si on veut la condition standard
            hmatrix_directional_builder.set_admissibility_condition(directional_admissibility);
            // hmatrix_directional_builder_inverse.set_admissibility_condition(directional_admissibility);
            // hmatrix_directional_lu_builder.set_admissibility_condition(directional_admissibility);

            //////////////////////////////
            // tests avec la matrice L/U
            // Matricegenerator<double, double> generator_lu(LU, *root_directional, *root_directional);
            // Matrix<double> generator_lu_dense(size, size);
            // generator_lu.copy_submatrix(size, size, 0, 0, generator_lu_dense.data());

            // HMatrix<double, double> hmatrix_lu = hmatrix_directional_lu_builder.build(generator_lu);
            // Matrix<double> hlu_dense(size, size);
            // copy_to_dense(hmatrix_lu, hlu_dense.data());
            // std::cout << "erreur H(LU)  " << normFrob(LU - hlu_dense) / normFrob(LU) << ',' << normFrob(hlu_dense) << ',' << normFrob(LU) << ',' << normFrob(generator_lu_dense) << std::endl;
            // std::cout << "compression " << hmatrix_lu.get_compression() << std::endl;
            /////////////////////////////////////
            // MATRICE QU ON DONNE A HTOOL
            // premier cas : la matrice direct deuxième cas l'inverse
            // Matricegenerator<double, double> generator_directional(reference_dense, *root_directional, *root_directional);
            // Matricegenerator<double, double> generator_lu(LU, *root_directional, *root_directional);
            // Matricegenerator<double, double> generator_inverse(vrai_mat, *root_directional, *root_directional);

            // CSR_generator<double, double> gen_csr(val, row, col, *root_directional, *root_directional);
            auto start = std::chrono::high_resolution_clock::now();

            // CSR_generator_random<double, double> gen_csr(*root_directional, *root_directional, row, col, val, def, 3);
            CSR_generator<double, double> gen_csr(*root_directional, *root_directional, row, col, val);
            // Matrix<double> csr_to_dense(size, size);
            // gen_csr.copy_submatrix(size, size, 0, 0, csr_to_dense.data());
            auto stop     = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            // std::cout << "Durée  CSR : " << duration.count() << " microsecondes" << std::endl;

            // std::cout << "gen_csr copy_sub ok" << std::endl;
            // Matrix<double> generator_dir_to_dense(size, size);
            // generator_directional.copy_submatrix(size, size, 0, 0, generator_dir_to_dense.data());
            // std::cout << "gen_dir_dense_sub ok " << std::endl;

            // std::cout << "erreur approche csr : " << normFrob(csr_to_dense - generator_dir_to_dense) << "|" << normFrob(csr_to_dense) << ',' << normFrob(generator_dir_to_dense) << std::endl;

            // std::cout << " H(LU) " << std::endl;
            // HMatrix<double, double> LU_h = hmatrix_directional_lu_builder.build(generator_lu);
            // Matrix<double> gen_lu_dense(size, size);
            // Matrix<double> LU_h_dense(size, size);
            // generator_lu.copy_submatrix(size, size, 0, 0, gen_lu_dense.data());
            // std::cout << "1   ok " << std::endl;
            // copy_to_dense(LU_h, LU_h_dense.data());
            // std::cout << "erreur H(LU)-LU : " << normFrob(LU_h_dense - gen_lu_dense) / normFrob(gen_lu_dense) << " et compression : " << LU_h.get_compression() << std::endl;
            // LU_h.save_plot("LU_h");
            //////////////////// build sans HTOOL
            // HMatrix<double, double> test_sans_htool(root_directional, root_directional);
            // test_sans_htool.set_admissibility_condition(directional_admissibility);
            // test_sans_htool.set_epsilon(epsilon);
            // test_sans_htool.set_eta(eta);
            // build_hmatrix(gen_csr, &test_sans_htool);
            // Matrix<double> test_dense(size, size);
            // copy_to_dense(test_sans_htool, test_dense.data());
            // std::cout << "erreur et compression sans htool : " << normFrob(test_dense - generator_dir_to_dense) / normFrob(generator_dir_to_dense) << " et une compression de : " << test_sans_htool.get_compression() << std::endl;

            // Matrix<double> generator_lu_to_dense(size, size);
            // generator_lu.copy_submatrix(size, size, 0, 0, generator_lu_to_dense.data());
            // Matrix<double> generator_inverse_to_dense(size, size);
            // generator_inverse.copy_submatrix(size, size, 0, 0, generator_inverse_to_dense.data());
            // On build la hmatrice
            // auto hmatrix_directional         = hmatrix_directional_builder.build(generator_directional);
            // auto hmatrix_directional_inverse = hmatrix_directional_builder_inverse.build(generator_inverse);
            std::cout << "build hmat " << std::endl;
            auto hmatrix_directional = hmatrix_directional_builder.build(gen_csr);
            std::cout << " build ok" << std::endl;
            hmatrix_directional.save_plot("hmatrix_directional");
            // auto hmatrix_directional_inverse = hmatrix_directional_builder_inverse.build(gen_csr);
            // on regarde la compression et l'erreur
            // Matrix<double> directional_dense(size, size);
            // // Matrix<double> directional_inverse_dense(size, size);
            // copy_to_dense(hmatrix_directional, directional_dense.data());
            // // copy_to_dense(hmatrix_directional_inverse, directional_inverse_dense.data());
            // std::cout << "erreur directional : " << normFrob(directional_dense - generator_dir_to_dense) / normFrob(reference_dense) << std::endl;
            // std::cout << " erreur sur l'inverse : " << normFrob(directional_inverse_dense - generator_inverse_to_dense) / normFrob(vrai_mat) << std::endl;
            std::cout << "compression directional : " << hmatrix_directional.get_compression() << std::endl;

            // // std::cout << " compression inverse : " << hmatrix_directional_inverse.get_compression() << std::endl;
            // // hmatrix_directional.set_eta(eta);
            // // std::cout << "ACA sur la matrice directrionnelle" << std::endl;
            // // HMatrix<double, double> aca_directional(*root_directional, *root_directional);
            // // hmatrix_directional.format_ACA(&aca_directional, 1e-6);
            // // Matrix<double> aca_directional_dense(size, size);
            // // copy_to_dense(aca_directional, aca_directional_dense.data());
            // // std::cout << "erreur approximation ACA de directionnelle : " << normFrob(aca_directional_dense - generator_dir_to_dense) / normFrob(generator_dir_to_dense) << std::endl;
            // // std::cout << " avec une compression de : " << aca_directional.get_compression() << std::endl;
            // ///////////////////////////////////
            // // FACTORISATION LU
            std::cout << "Factoriation LU" << std::endl;
            auto x_random = generate_random_vector(size);
            Matrix<double> X_random(size, 1);
            X_random.assign(size, 1, x_random.data(), false);
            // std::vector<double> x_random(X_random.data(), X_random.data() + size);

            // auto Ax_directional = generator_dir_to_dense * X_random;
            std::vector<double> ax_directional(size);
            gen_csr.mat_vect(x_random, ax_directional);
            Matrix<double> Ax_directional(size, 1);
            for (int k = 0; k < size; ++k) {
                Ax_directional(k, 0) = ax_directional[k];
            }
            // hmatrix_directional.set_eta(eta);
            // hmatrix_directional.set_epsilon(epsilon);
            std::cout << "lu start " << std::endl;
            HMatrix<double> Lh(*root_directional, *root_directional);
            Lh.set_admissibility_condition(directional_admissibility);
            Lh.set_epsilon(epsilon);
            Lh.set_eta(eta);
            Lh.set_low_rank_generator(hmatrix_directional.get_low_rank_generator());

            HMatrix<double, double> Uh(*root_directional, *root_directional);
            Uh.set_admissibility_condition(directional_admissibility);
            Uh.set_epsilon(epsilon);
            Uh.set_eta(eta);
            Uh.set_low_rank_generator(hmatrix_directional.get_low_rank_generator());

            start = std::chrono::high_resolution_clock::now();
            lu_factorization(hmatrix_directional);
            hmatrix_directional.save_plot("hlu_directional");
            lu_solve('N', hmatrix_directional, Ax_directional);
            // HLU_fast(hmatrix_directional, *root_directional, &Lh, &Uh);

            // auto res = hmatrix_directional.solve_LU_triangular(Lh, Uh, ax_directional);
            stop = std::chrono::high_resolution_clock::now();
            std::cout << "lu ok " << std::endl;
            auto duration_lu = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
            // for (auto &l : hmatrix_directional.get_leaves()) {
            //     if (l->is_low_rank()) {
            //         auto U = l->get_low_rank_data()->get_U();
            //         auto V = l->get_low_rank_data()->get_V();
            //         std::cout << "feuille lr de taille " << l->get_target_cluster().get_size() << ',' << l->get_source_cluster().get_size() << " et de norme : " << normFrob(U * V) << std::endl;
            //     }
            // }

            /// test pour vérifier que le HLU compresse vraiment bien

            // Matrix<double> lu_dense(size, size);
            // copy_to_dense(hmatrix_directional, lu_dense.data());
            // auto lu_unperm = get_unperm_mat(lu_dense, *root_directional, *root_directional);
            // // Matricegenerator_noperm<double, double> lu_gen(lu_dense, *root_directional, *root_directional);
            // Matricegenerator<double, double> lu_gen(lu_unperm, *root_directional, *root_directional);
            // Matrix<double> gen_lu_dense(size, size);
            // lu_gen.copy_submatrix(size, size, 0, 0, gen_lu_dense.data());
            // std::cout << "generator lu : " << normFrob(gen_lu_dense - lu_dense) / normFrob(lu_dense) << std::endl;
            // HMatrixTreeBuilder<double, double> hmatrix_directional_lu_builder(*root_directional, *root_directional, epsilon, 2.5, 'N', 'N', -1, -1, -1);
            // // std::shared_ptr<Bandee<double>> directional_admissibility = std::make_shared<Bandee<double>>(bb, def);
            // hmatrix_directional_lu_builder.set_admissibility_condition(directional_admissibility);
            // HMatrix<double, double> lu_h = hmatrix_directional_lu_builder.build(lu_gen);
            // std::cout << "build ok " << std::endl;
            // Matrix<double> lu_h_dense(size, size);
            // copy_to_dense(lu_h, lu_h_dense.data());
            // std::cout << "erreur lu_ h : " << normFrob(lu_h_dense - lu_dense) / normFrob(lu_dense) << "   et compression : " << lu_h.get_compression() << std::endl;
            // lu_h.save_plot("LU_h");
            // std::cout << "ACA begin " << std::endl;
            // HMatrix<double, double>
            //     aca_directional(*root_directional, *root_directional);
            // hmatrix_directional.format_ACA(&aca_directional, 1e-6);
            // std::cout << "ACA ok " << std::endl;
            // hmatrix_directional.save_plot("hmatrix_lu");
            // aca_directional.save_plot("aca_lu");
            // std::cout << "compression aca lu : " << aca_directional.get_compression() << std::endl;
            std::cout
                << "compression du LU directional : " << hmatrix_directional.get_compression() << std::endl;
            // lu_factorization(hmatrix_directional_inverse);
            // std::cout << "compression du LU inverse : " << hmatrix_directional_inverse.get_compression() << std::endl;
            // lu_factorization(aca_directional);
            // std::cout << "compression du LU ACA directional : " << aca_directional.get_compression() << std::endl;
            // lu_factorization(hmatrix_directional_inverse);
            // std::cout << "compression du LU inverse : " << hmatrix_directional_inverse.get_compression() << std::endl;

            // auto AX_inverse     = generator_inverse_to_dense * X_random;

            // lu_solve('N', aca_directional, Ax_directional);

            std::cout << "-------------------------->erreur solve lu directional : " << normFrob(Ax_directional - X_random) / normFrob(X_random) << std::endl;
            // std::cout << "-------------------------->erreur solve lu directional : " << norm2(res - x_random) / norm2(x_random) << std::endl;

            std::cout << "-------------------------->time lu et solve : " << duration_lu.count() << std::endl;

            // lu_solve('N', hmatrix_directional_inverse, AX_inverse);
            // std::cout << "erreur solve lu inverse : " << normFrob(AX_inverse - X_random) / normFrob(X_random) << std::endl;
            ////////////////////////////////////
        }
        std::cout
            << "fini" << std::endl;
    }
    return 0;
}